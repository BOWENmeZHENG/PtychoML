from pathlib import Path
from ptychoml.data.objects import prepare_objects, save_objects
from ptychoml.data.simulate import simulate_cxi_batch
import torch
import numpy as np

DATA_DIR = Path(__file__).parents[3] / "data/raw/BBBC002/drosophila_kc167_1_images"
SAVE_DIR = Path(__file__).parents[3] / "data/processed/BBBC002"
GEO_CXI_PATH = Path(__file__).parents[3] / "data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
PROBE_PATH = Path(__file__).parents[3] / "data/raw/probes/probe_data_gen_104.npy"
SEED = 123
CROPS_PER_IMAGE = 4
CROP_SIZE = 200


def main():
    # Prepare complex objects from the raw image dataset and save
    objects, source = prepare_objects(DATA_DIR, CROPS_PER_IMAGE, CROP_SIZE, SEED)
    save_objects(SAVE_DIR / "objects.npz", objects, source)
    print(f"Prepared {len(objects)} complex objects from {DATA_DIR} and saved to {SAVE_DIR / 'objects.npz'}")

    # Simulate .cxi file
    objects = torch.as_tensor(objects)
    probe = torch.as_tensor(np.load(PROBE_PATH))
    # test simulation
    simulate_cxi_batch(objects, SAVE_DIR / "simulated", GEO_CXI_PATH, probe)


if __name__ == "__main__":
    main()