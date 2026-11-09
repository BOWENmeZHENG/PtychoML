"""Reconstruct one .cxi scan with a cdtools model to inspect the result (for training data, use ptychoml.recon.batch).

Saves the cdtools .h5 and amplitude/phase PNGs, plus a summary figure with the loss curve and, for simulated scans, the
ground truth next to the reconstruction. Use ``--snapshot-at`` to also save earlier iterations of the same run.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np

from ptychoml.recon.core import (
    OUT_DIR,
    ROOT_DIR,
    add_recon_args,
    footprint_object,
    iter_reconstruction,
    load_ground_truth,
)

logger = logging.getLogger(__name__)

CXI = ROOT_DIR / "data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
SINGLE_DIR = OUT_DIR / "single_recon"  # keeps these inspection runs apart from the batch training data


def save_results(model, h5_path, obj_padding, plt):
    """Save the model to h5_path, plus object amplitude and phase PNGs next to it."""
    model.save_to_h5(h5_path)
    obj = footprint_object(model, obj_padding)
    for name, image, cmap in (("amplitude", np.abs(obj), "viridis"), ("phase", np.angle(obj), "twilight")):
        plt.imsave(h5_path.with_name(f"{h5_path.stem}_obj_{name}.png"), image, cmap=cmap)


def save_summary(model, cxi, png_path, obj_padding, plt):
    """Save a figure of the reconstruction (and ground truth, if the scan has one) above the loss curve."""
    views = [("reconstruction", footprint_object(model, obj_padding))]
    gt = load_ground_truth(cxi)
    if gt is not None:
        views.append(("ground truth", gt))

    fig = plt.figure(figsize=(8, 4 * (len(views) + 1)), layout="constrained")
    grid = fig.add_gridspec(len(views) + 1, 2)
    for row, (title, obj) in enumerate(views):
        for col, (name, image, cmap) in enumerate(
            (("amplitude", np.abs(obj), "viridis"), ("phase", np.angle(obj), "twilight"))
        ):
            ax = fig.add_subplot(grid[row, col])
            fig.colorbar(ax.imshow(image, cmap=cmap), ax=ax, shrink=0.8)
            ax.set_title(f"{title} {name}")
            ax.axis("off")
    loss_ax = fig.add_subplot(grid[-1, :])
    loss_ax.semilogy(model.loss_history)
    loss_ax.set(xlabel="iteration", ylabel="loss", title="loss")
    fig.savefig(png_path)


def main(argv: list[str] | None = None) -> None:
    """Run the reconstruction; argv defaults to ``sys.argv[1:]`` when ``None``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxi", type=Path, default=CXI, help="Path to the .cxi file")
    add_recon_args(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output .h5 path (default: {SINGLE_DIR.name}/<cxi name>_<model>/<cxi name>_<model>_iter_<iterations>.h5 "
        f"under {OUT_DIR})",
    )
    parser.add_argument(
        "--snapshot-at",
        type=int,
        nargs="+",
        default=[],
        metavar="ITER",
        help="Also save a reconstruction after these iterations, e.g. '--snapshot-at 5' (same run, so same gauge)",
    )
    parser.add_argument("--show", action="store_true", help="Open the summary figure in a window when done")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(not 1 <= n <= args.iterations for n in args.snapshot_at):
        parser.error(f"--snapshot-at values must be between 1 and --iterations ({args.iterations})")

    if not args.show:
        matplotlib.use("Agg")  # headless: only write files
    from matplotlib import pyplot as plt

    prefix = f"{args.cxi.stem}_{args.model}"
    out_path = args.out or SINGLE_DIR / prefix / f"{prefix}_iter_{args.iterations}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch, model in iter_reconstruction(
        args.cxi, args.model, args.iterations, args.lr, args.batch_size, args.obj_padding, args.device
    ):
        logger.info(model.report())
        if epoch in args.snapshot_at:
            snapshot_path = out_path.parent / f"{prefix}_iter_{epoch}.h5"
            save_results(model, snapshot_path, args.obj_padding, plt)
            logger.info("Saved iteration %d snapshot to %s", epoch, snapshot_path)

    save_results(model, out_path, args.obj_padding, plt)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.png")
    save_summary(model, args.cxi, summary_path, args.obj_padding, plt)
    logger.info("Saved reconstruction to %s and summary to %s", out_path, summary_path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
