import argparse
from pathlib import Path

import cdtools
from matplotlib import pyplot as plt

from ptychoml.cdtools_ext import MLAugmentedPtycho

CXI = Path(__file__).parents[1] / "data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
MODELS = {
    "standard": cdtools.models.FancyPtycho,
    "ml": MLAugmentedPtycho,
}
ITERATIONS = 100
LR = 0.01
BS = 50
OUT_DIR = Path(__file__).parents[1] / "data/toy"
OUT_FILENAME = "result.h5"

def main():
    parser = argparse.ArgumentParser(description="Run cdtools with specified model and .cxi file")
    parser.add_argument("--cxi", type=Path, default=CXI, help="Path to the .cxi file")
    parser.add_argument("--model", choices=MODELS, default="standard", help="Model to use, either 'standard' or 'ml'")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BS, help="Batch size")
    args = parser.parse_args()

    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(args.cxi))
    # dataset.get_as(device='mps')

    model = MODELS[args.model].from_dataset(dataset)
    # model.to(device='mps')

    for _loss in model.Adam_optimize(args.iterations, dataset, lr=args.lr, batch_size=args.batch_size):
        print(model.report())

    model.inspect(dataset)
    model.compare(dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_to_h5(OUT_DIR / OUT_FILENAME)

    plt.show()


if __name__ == "__main__":
    main()
