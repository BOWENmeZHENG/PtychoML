import argparse
from pathlib import Path

import cdtools
import matplotlib

from ptychoml.cdtools_ext import MLAugmentedPtycho

CXI = Path(__file__).parents[1] / "data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
MODELS = {
    "standard": cdtools.models.FancyPtycho,
    "ml": MLAugmentedPtycho,
}
ITERATIONS = 50
LR = 0.01
BS = 50
OBJ_PADDING = 20
OUT_DIR = Path(__file__).parents[1] / "data/recon"


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a .cxi scan with a cdtools model")
    parser.add_argument("--cxi", type=Path, default=CXI, help="Path to the .cxi file")
    parser.add_argument("--model", choices=MODELS, default="standard", help="Model to use, either 'standard' or 'ml'")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BS, help="Batch size")
    parser.add_argument("--obj-padding", type=int, default=OBJ_PADDING, help="Object padding in pixels around the scanned area")
    parser.add_argument("--device",default=None, help="Torch device, e.g. 'cuda' or 'mps' (default: CPU)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output .h5 path (default: {OUT_DIR.name}/<cxi name>_<model>.h5 under {OUT_DIR.parent})",
    )
    parser.add_argument(
        "--snapshot-at",
        type=int,
        nargs="+",
        default=[],
        metavar="ITER",
        help="Also save a reconstruction after these iterations, e.g. '--snapshot-at 5' (same run, so same gauge)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (for headless or batch runs)")
    args = parser.parse_args()
    if any(not 1 <= n <= args.iterations for n in args.snapshot_at):
        parser.error(f"--snapshot-at values must be between 1 and --iterations ({args.iterations})")

    if args.no_plot:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    prefix = f"{args.cxi.stem}_{args.model}"
    out_path = args.out or OUT_DIR / f"{prefix}_iter_{args.iterations}.h5"

    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(args.cxi))
    model = MODELS[args.model].from_dataset(dataset, obj_padding=args.obj_padding)
    if args.device:
        dataset.get_as(device=args.device)
        model.to(device=args.device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One Adam_optimize call for the whole run: calling it twice would reset the optimizer state.
    for epoch, _loss in enumerate(
        model.Adam_optimize(args.iterations, dataset, lr=args.lr, batch_size=args.batch_size), start=1
    ):
        print(model.report())
        if epoch in args.snapshot_at:
            snapshot_path = out_path.parent / f"{prefix}_iter_{epoch}.h5"
            model.save_to_h5(snapshot_path)
            print(f"Saved iteration {epoch} snapshot to {snapshot_path}")

    model.save_to_h5(out_path)
    print(f"Saved reconstruction to {out_path}")

    if not args.no_plot:
        model.inspect(dataset)
        model.compare(dataset)
        plt.show()


if __name__ == "__main__":
    main()
