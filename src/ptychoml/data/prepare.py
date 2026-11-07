"""Prepare complex objects from raw images and simulate one .cxi scan per object."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ptychoml.data.objects import prepare_objects, save_objects
from ptychoml.data.simulate import simulate_cxi_batch

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[3]
RAW_DATA_DIR = ROOT_DIR / "data/raw/BBBC002/drosophila_kc167_1_images"
GEO_CXI_PATH = ROOT_DIR / "data/raw/geometries/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
PROBE_PATH = ROOT_DIR / "data/raw/probes/probe_data_gen_104.npy"
SAVE_DIR = ROOT_DIR / "data/processed/BBBC002"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Load the ``prepare`` section of params.yaml, which DVC tracks for the prepare stage."""
    with open(path) as f:
        return yaml.safe_load(f)["prepare"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR, help="Directory of raw images")
    parser.add_argument("--geo-cxi-path", type=Path, default=GEO_CXI_PATH, help="Geometry .cxi file to simulate with")
    parser.add_argument("--probe-path", type=Path, default=PROBE_PATH, help="Probe .npy file")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR, help="Output directory")
    parser.add_argument("--crops-per-image", type=int, default=params["crops_per_image"], help="Random crops per image")
    parser.add_argument("--crop-size", type=int, default=params["crop_size"], help="Crop side length in pixels")
    parser.add_argument("--seed", type=int, default=params["seed"], help="Random seed")
    parser.add_argument("--step", type=int, default=params["step"], help="Scan step between probe positions in pixels")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the data preparation pipeline.

    Takes random crops from the raw images and makes them complex-valued, then
    simulates one scan per object. Writes to ``--save-dir``:

    - ``objects.npz``: the complex objects, their source-image indices and the
      crop parameters (``m``, ``N``, ``seed``).
    - ``simulated/sim_*.cxi``: one simulated .cxi file per object.

    argv: argument list to parse; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Prepare complex objects from the raw image dataset and save
    objects, source = prepare_objects(args.raw_data_dir, args.crops_per_image, args.crop_size, args.seed)
    objects_path = args.save_dir / "objects.npz"
    save_objects(objects_path, objects, source, m=args.crops_per_image, N=args.crop_size, seed=args.seed)
    logger.info("Prepared %d complex objects from %s and saved to %s", len(objects), args.raw_data_dir, objects_path)

    # Simulate .cxi files and save
    objects = torch.as_tensor(objects)
    probe = torch.as_tensor(np.load(args.probe_path))
    simulate_cxi_batch(objects, args.save_dir / "simulated", args.geo_cxi_path, probe, step=args.step)


if __name__ == "__main__":
    main()
