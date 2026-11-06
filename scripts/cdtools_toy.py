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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxi", type=Path, default=CXI, help="Path to the .cxi file")
    parser.add_argument("--model", choices=MODELS, default="standard", help="Model to use, either 'standard' or 'ml'")
    args = parser.parse_args()

    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(args.cxi))
    # dataset.get_as(device='mps')

    model = MODELS[args.model].from_dataset(dataset)
    # model.to(device='mps')

    for _loss in model.Adam_optimize(100, dataset, lr=0.01, batch_size=50):
        print(model.report())

    model.inspect(dataset)
    model.compare(dataset)
    model.save_to_h5(Path(__file__).parents[1] / "data/toy/result.h5")

    plt.show()


if __name__ == "__main__":
    main()
