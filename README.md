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
| `--crops-per-image` | `prepare.crops_per_image` in `params.yaml` (`4`) | Random crops taken per image |
| `--crop-size` | `prepare.crop_size` in `params.yaml` (`200`) | Crop side length in pixels |
| `--seed` | `prepare.seed` in `params.yaml` (`123`) | Random seed |

The `prepare` stage in `dvc.yaml` runs this script with the values from `params.yaml`. `uv run dvc repro` regenerates `data/processed/BBBC002` whenever `data/raw`, the three `ptychoml/data` modules or those params change. Edit `params.yaml` rather than passing flags if you want DVC to track the change.

Outputs in `--save-dir`:

- `objects.npz`: complex objects (`crops`), the index of the source image for each (`source`, useful for splitting train/validation by file), and the crop parameters (`m`, `N`, `seed`).
- `simulated/sim_*.cxi`: one simulated scan per object.

The raw images come from the [BBBC002](https://bbbc.broadinstitute.org/BBBC002) Drosophila Kc167 cell dataset.

## Reconstruction

`scripts/cdtools_recon.py` reconstructs any `.cxi` scan (the toy gold-ball scan in `data/toy/` by default) with either the standard cdtools model or `MLAugmentedPtycho`:

```bash
uv run python scripts/cdtools_recon.py --model standard   # or: --model ml
uv run python scripts/cdtools_recon.py --cxi data/processed/BBBC002/simulated/sim_0.cxi --device mps --no-plot
```

Options: `--cxi`, `--model`, `--iterations`, `--lr`, `--batch-size`, `--device`, `--out`, `--no-plot` (see `--help`). The result is saved to `--out`, by default `data/recon/<cxi name>_<model>.h5`.

## Project layout

```
src/ptychoml/
  data/            # object preparation (objects.py), scan simulation (simulate.py), pipeline entry point (prepare.py)
  cdtools_ext/     # cdtools model/tool extensions
scripts/           # standalone scripts
tests/             # pytest unit tests
data/              # raw, processed and toy data (large files tracked with DVC)
dvc.yaml           # DVC pipeline (prepare stage)
params.yaml        # parameters for the prepare stage
```

## Development

```bash
uv run pytest      # tests
uv run ruff check  # lint
uv run ruff format # format
```
