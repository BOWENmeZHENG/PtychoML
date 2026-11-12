"""Training data: pairs of an early and a late reconstruction of the same scan, split by source image.

``ptychoml.recon.batch`` saves one ``recon_<i>.npy`` per scan with the object after several iterations, listed in the
folder's ``meta.json``. Here the object at ``input_iter`` is paired with the object at ``target_iter``. The two
come from the same run and share a gauge, so no alignment is needed. Complex values become two real channels (real,
imaginary) with ``complex_to_2ch``.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ptychoml.cdtools_ext.tools import complex_to_2ch
from ptychoml.recon.core import PROCESSED_DIR

__all__ = [
    "TrainingData",
    "compute_scale",
    "load_pairs",
    "load_training_data",
    "snapshot_positions",
    "split_by_source",
]

META = "meta.json"  # written by ptychoml.recon.batch
OBJECTS = "objects.npz"  # written by ptychoml.data.prepare, has the source image of every crop
RECON_FILE = re.compile(r"recon_(\d+)\.npy")  # not recon_<i>.tmp.npy, the half-written files of an interrupted run


def snapshot_positions(meta: dict, input_iter: int, target_iter: int) -> tuple[int, int]:
    """Positions of ``input_iter`` and ``target_iter`` along the first axis of a ``recon_*.npy`` file."""
    iterations = meta["iterations"]
    missing = [n for n in (input_iter, target_iter) if n not in iterations]
    if missing:
        raise ValueError(f"iterations {missing} were not saved, the recon folder has {iterations}")
    return iterations.index(input_iter), iterations.index(target_iter)


def load_pairs(
    recon_dir: str | Path, input_iter: int, target_iter: int, processed_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the (input, target) objects of every reconstructed scan in ``recon_dir``.

    Args:
        recon_dir: Output folder of ``ptychoml.recon.batch``.
        input_iter: Iteration of the input object.
        target_iter: Iteration of the target object.
        processed_dir: Folder with ``objects.npz``. Default: the processed folder named in ``meta.json``.

    Returns:
        ``(x, y, source)``: complex64 arrays of shape (N, H, W) and the source image index of each of the N pairs.
    """
    recon_dir = Path(recon_dir)
    meta = json.loads((recon_dir / META).read_text())
    k_in, k_out = snapshot_positions(meta, input_iter, target_iter)
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR / meta["data_dir"]
    with np.load(processed_dir / OBJECTS) as f:
        all_sources = f["source"]

    files = sorted((int(m[1]), p) for p in recon_dir.iterdir() if (m := RECON_FILE.fullmatch(p.name)))
    if not files:
        raise FileNotFoundError(f"no recon_*.npy files in {recon_dir}")
    indices = np.array([i for i, _ in files])
    if indices.max() >= len(all_sources):
        raise ValueError(f"{recon_dir} has recon_{indices.max()} but {processed_dir / OBJECTS} has {len(all_sources)}")
    arrays = [np.load(p, mmap_mode="r") for _, p in files]
    x = np.stack([a[k_in] for a in arrays])
    y = np.stack([a[k_out] for a in arrays])
    return x, y, all_sources[indices]


def split_by_source(source: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into train and validation so that no source image is in both.

    Crops of one microscopy image look alike, so splitting by crop would leak the validation data into training.
    Whole source images are drawn at random: at least one for validation and at least one for training.
    """
    sources = np.unique(source)
    if len(sources) < 2:
        raise ValueError("need at least two source images to split into train and validation")
    n_val = min(max(1, round(val_fraction * len(sources))), len(sources) - 1)
    val_sources = np.random.default_rng(seed).permutation(sources)[:n_val]
    is_val = np.isin(source, val_sources)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def compute_scale(x: np.ndarray) -> float:
    """RMS of the real and imaginary parts of ``x``, the factor that brings the training inputs to unit scale."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2) / 2))


def to_tensor(a: np.ndarray, scale: float) -> torch.Tensor:
    """Complex (N, H, W) array to a float32 (N, 2, H, W) tensor of real and imaginary parts, divided by ``scale``."""
    return complex_to_2ch(torch.from_numpy(np.ascontiguousarray(a))) / scale


@dataclass
class TrainingData:
    """Train and validation sets of (input, target) tensors of shape (2, H, W), and the scale used on them.

    ``scale`` divides both inputs and targets. ``train_idx`` and ``val_idx`` index the pairs in ``recon_dir`` order.
    """

    train: TensorDataset
    val: TensorDataset
    scale: float
    image_shape: tuple[int, int]
    train_idx: np.ndarray
    val_idx: np.ndarray


def load_training_data(
    recon_dir: str | Path,
    input_iter: int,
    target_iter: int,
    val_fraction: float = 0.2,
    seed: int = 0,
    processed_dir: str | Path | None = None,
) -> TrainingData:
    """Load the pairs, split them by source image, scale them by the training inputs' RMS and wrap them as datasets.

    The scale comes from the training inputs alone, so the validation data does not influence it.
    """
    x, y, source = load_pairs(recon_dir, input_iter, target_iter, processed_dir)
    train_idx, val_idx = split_by_source(source, val_fraction, seed)
    scale = compute_scale(x[train_idx])
    x_t, y_t = to_tensor(x, scale), to_tensor(y, scale)
    return TrainingData(
        train=TensorDataset(x_t[train_idx], y_t[train_idx]),
        val=TensorDataset(x_t[val_idx], y_t[val_idx]),
        scale=scale,
        image_shape=(x.shape[1], x.shape[2]),
        train_idx=train_idx,
        val_idx=val_idx,
    )
