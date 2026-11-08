"""Shared pieces of the reconstruction command-line tools: models, defaults, the optimizer loop and state extraction."""

import argparse
from collections.abc import Iterator
from pathlib import Path

import cdtools
import h5py
import numpy as np

from ptychoml.cdtools_ext import MLAugmentedPtycho

__all__ = [
    "BS",
    "GT_OBJ",
    "ITERATIONS",
    "LR",
    "MODELS",
    "OBJ_PADDING",
    "OUT_DIR",
    "PROCESSED_DIR",
    "add_recon_args",
    "footprint_object",
    "iter_reconstruction",
    "load_ground_truth",
    "probe_array",
]

ROOT_DIR = Path(__file__).parents[3]
PROCESSED_DIR = ROOT_DIR / "data/processed"
OUT_DIR = ROOT_DIR / "data/recon"
MODELS = {
    "standard": cdtools.models.FancyPtycho,
    "ml": MLAugmentedPtycho,
}
ITERATIONS = 100
LR = 0.01
BS = 50
OBJ_PADDING = 20
GT_OBJ = "entry_1/image_1/data"  # ground-truth object written by ptychoml.data.simulate


def add_recon_args(parser: argparse.ArgumentParser) -> None:
    """Add the options that configure one reconstruction run."""
    parser.add_argument("--model", choices=MODELS, default="standard", help="Model to use, either 'standard' or 'ml'")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BS, help="Batch size")
    parser.add_argument(
        "--obj-padding", type=int, default=OBJ_PADDING, help="Object padding in pixels around the scanned area"
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. 'cuda' or 'mps' (default: CPU)")


def iter_reconstruction(
    cxi: str | Path,
    model_name: str = "standard",
    iterations: int = ITERATIONS,
    lr: float = LR,
    batch_size: int = BS,
    obj_padding: int = OBJ_PADDING,
    device: str | None = None,
) -> Iterator[tuple[int, cdtools.models.CDIModel]]:
    """Reconstruct ``cxi`` and yield ``(epoch, model)`` after each iteration, with epochs counted from 1.

    The same model object is yielded every time, so copy whatever you need before advancing.
    """
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(cxi))
    model = MODELS[model_name].from_dataset(dataset, obj_padding=obj_padding)
    if device:
        dataset.get_as(device=device)
        model.to(device=device)
    # One Adam_optimize call for the whole run: calling it twice would reset the optimizer state.
    for epoch, _loss in enumerate(model.Adam_optimize(iterations, dataset, lr=lr, batch_size=batch_size), start=1):
        yield epoch, model


def footprint_object(model: cdtools.models.CDIModel, obj_padding: int) -> np.ndarray:
    """Complex64 (H, W) first object mode, cropped to the full scanned footprint (positions plus probe size).

    Removing the padding gives the same shape as the ground truth. ``model.obj_view_slice`` is smaller: it only spans
    the probe centres.
    """
    obj = model.obj.detach().cpu()
    if obj.dim() > 2:
        obj = obj[0]
    if obj_padding:
        obj = obj[obj_padding:-obj_padding, obj_padding:-obj_padding]
    return obj.numpy().astype(np.complex64)


def probe_array(model: cdtools.models.CDIModel) -> np.ndarray:
    """Complex64 (modes, h, w) probe of the model."""
    return model.probe.detach().cpu().numpy().astype(np.complex64)


def load_ground_truth(cxi: str | Path) -> np.ndarray | None:
    """Complex64 ground-truth object stored in a simulated .cxi, or ``None`` for real data."""
    with h5py.File(cxi) as f:
        return f[GT_OBJ][()].astype(np.complex64) if GT_OBJ in f else None
