"""Generate ML training data: reconstruct every simulated .cxi in a processed-data folder and save snapshots.

Each scan gets one ``recon_*.npy`` of shape (K, H, W), the complex64 object after each of the K snapshot iterations
(all from the same run, so the same gauge), amplitude and phase PNGs of each snapshot next to it, and the folder gets
one ``meta.json`` with the iterations and settings.
The files are uncompressed so they can be memory-mapped: ``np.load(path, mmap_mode="r")``. The ground truth is not
copied here; it is in ``objects.npz`` of the processed folder (``recon_i`` matches ``sim_i``). To inspect a single
reconstruction, use ``ptychoml.recon.single`` instead.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib.image import imsave
from tqdm import tqdm

from ptychoml.recon.core import (
    OBJ_PADDING,
    OUT_DIR,
    PROCESSED_DIR,
    add_recon_args,
    footprint_object,
    iter_reconstruction,
)

logger = logging.getLogger(__name__)

SNAPSHOTS = [5, 50]
META = "meta.json"
TRAINING_DIR = OUT_DIR / "training"  # keeps the batch training data apart from the single_recon inspection runs


def reconstruct(cxi, snapshot_at, obj_padding=OBJ_PADDING, **kwargs):
    """Reconstruct one scan and return the (K, H, W) complex64 objects after each iteration in ``snapshot_at``.

    kwargs are passed to ``iter_reconstruction``.
    """
    objs = [
        footprint_object(model, obj_padding)
        for epoch, model in iter_reconstruction(cxi, obj_padding=obj_padding, **kwargs)
        if epoch in snapshot_at
    ]
    return np.stack(objs)


def save_images(objs, snapshot_at, prefix):
    """Save amplitude and phase PNGs of each snapshot as ``<prefix>_iter_<n>_obj_<amplitude|phase>.png``."""
    for epoch, obj in zip(snapshot_at, objs, strict=True):
        for name, image, cmap in (("amplitude", np.abs(obj), "viridis"), ("phase", np.angle(obj), "twilight")):
            imsave(prefix.with_name(f"{prefix.name}_iter_{epoch}_obj_{name}.png"), image, cmap=cmap)


def main(argv: list[str] | None = None) -> None:
    """Run the batch reconstruction; argv defaults to ``sys.argv[1:]`` when ``None``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help=f"Folder under {PROCESSED_DIR.name}/ with a simulated/ subfolder")
    add_recon_args(parser)
    parser.add_argument(
        "--snapshot-at",
        type=int,
        nargs="+",
        default=SNAPSHOTS,
        metavar="ITER",
        help="Iterations to save (the final one is always saved)",
    )
    parser.add_argument("--start", type=int, default=0, help="Index of the first scan to reconstruct")
    parser.add_argument("--stop", type=int, default=None, help="Index after the last scan (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Redo scans whose output already exists")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output folder (default: {OUT_DIR.name}/{TRAINING_DIR.name}/<data folder>_<model>_iter_<iterations>)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(not 1 <= n <= args.iterations for n in args.snapshot_at):
        parser.error(f"--snapshot-at values must be between 1 and --iterations ({args.iterations})")
    args.snapshot_at = sorted({*args.snapshot_at, args.iterations})  # always keep the final iteration

    data_dir = args.data_dir if args.data_dir.exists() else PROCESSED_DIR / args.data_dir
    cxis = sorted((data_dir / "simulated").glob("*.cxi"))[args.start : args.stop]
    out_dir = args.out_dir or TRAINING_DIR / f"{data_dir.name}_{args.model}_iter_{args.iterations}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reconstructing %d scans from %s into %s", len(cxis), data_dir, out_dir)

    # Files from different settings must not mix in one folder, so a resumed run has to match the earlier one.
    meta = {
        "data_dir": data_dir.name,
        "model": args.model,
        "iterations": args.snapshot_at,
        "total_iterations": args.iterations,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "obj_padding": args.obj_padding,
    }
    meta_path = out_dir / META
    if meta_path.exists() and not args.overwrite and json.loads(meta_path.read_text()) != meta:
        parser.error(f"{meta_path} has different settings; use a new --out-dir or pass --overwrite")
    meta_path.write_text(json.dumps(meta, indent=2))

    for cxi in tqdm(cxis, desc="Reconstructing", unit="scan"):
        out_path = out_dir / f"{cxi.stem.replace('sim_', 'recon_')}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        torch.manual_seed(0)
        objs = reconstruct(
            cxi,
            args.snapshot_at,
            model_name=args.model,
            iterations=args.iterations,
            lr=args.lr,
            batch_size=args.batch_size,
            obj_padding=args.obj_padding,
            device=args.device,
        )
        # Write then rename, so an interrupted run never leaves a half-written file that a re-run would skip.
        tmp_path = out_path.with_suffix(".tmp.npy")
        np.save(tmp_path, objs)
        save_images(objs, args.snapshot_at, out_path.with_suffix(""))
        tmp_path.rename(out_path)  # last, so an existing npy means the scan's PNGs are complete too


if __name__ == "__main__":
    main()
