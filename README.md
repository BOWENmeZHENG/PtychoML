# PtychoML

Machine-learning-assisted ptychographic reconstruction built on [cdtools](https://github.com/cdtools-developers/cdtools).

The project currently provides:

- a **simulated-data pipeline** that turns microscopy images into complex-valued objects and simulates one ptychography scan (`.cxi`) per object, and
- **cdtools extensions** (`MLAugmentedPtycho`) for experimenting with ML-modified reconstruction models.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data preparation

`ptychoml.data.prepare` builds a simulated dataset from raw images:

1. Takes `--crops-per-image` random `--crop-size`×`--crop-size` crops from every `.tif` in the raw image directory, each rotated by a random multiple of 90°. Crops that are essentially background are dropped.
2. Converts each crop to a complex transmission function (phase and absorption both scale with image intensity).
3. Simulates a raster scan of each object with a cdtools `FancyPtycho` forward model, using the detector geometry from an existing `.cxi` and a probe from a `.npy` file.

```bash
uv run python -m ptychoml.data.prepare
```

Defaults point at the layout below; every path can be overridden (see `--help`).

| Option | Default | Description |
| --- | --- | --- |
| `--raw-data-dir` | `data/raw/BBBC002/drosophila_kc167_1_images` | Directory of raw `.tif` images |
| `--geo-cxi-path` | `data/raw/geometries/AuBalls_700ms_30nmStep_3_6SS_filter.cxi` | Geometry `.cxi` to simulate with |
| `--probe-path` | `data/raw/probes/probe_data_gen_104.npy` | Probe `.npy` file |
| `--save-dir` | `data/processed/BBBC002` | Output directory |
| `--crops-per-image` | `4` | Random crops taken per image |
| `--crop-size` | `200` | Crop side length in pixels |
| `--seed` | `123` | Random seed |

Outputs in `--save-dir`:

- `objects.npz`: complex objects (`crops`), the index of the source image for each (`source`, useful for splitting train/validation by file), and the crop parameters (`m`, `N`, `seed`).
- `simulated/sim_*.cxi`: one simulated scan per object.

The raw images come from the [BBBC002](https://bbbc.broadinstitute.org/BBBC002) Drosophila Kc167 cell dataset.

## Toy reconstruction

`scripts/cdtools_toy.py` reconstructs the toy gold-ball scan in `data/toy/` with either the standard cdtools model or `MLAugmentedPtycho`:

```bash
uv run python scripts/cdtools_toy.py --model standard   # or: --model ml
```

The result is saved to `data/toy/result.h5`.

## Project layout

```
src/ptychoml/
  data/            # object preparation (objects.py), scan simulation (simulate.py), pipeline entry point (prepare.py)
  cdtools_ext/     # cdtools model/tool extensions
scripts/           # standalone scripts
tests/             # pytest unit tests
data/              # raw, processed and toy data (large files tracked with DVC)
dvc.yaml           # DVC pipeline (stages to be added)
```

## Development

```bash
uv run pytest      # tests
uv run ruff check  # lint
uv run ruff format # format
```
