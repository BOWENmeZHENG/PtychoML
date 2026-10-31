from pathlib import Path

import cdtools
from matplotlib import pyplot as plt

CXI = Path(__file__).parents[1] / "data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"


def main():
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(CXI))
    # dataset.get_as(device='mps')

    model = cdtools.models.FancyPtycho.from_dataset(dataset)
    # model.to(device='mps')

    for loss in model.Adam_optimize(10, dataset, lr=0.02, batch_size=10):
        print(model.report())

    model.inspect(dataset)
    model.compare(dataset)

    plt.show()


if __name__ == "__main__":
    main()
