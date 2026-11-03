"""Prepare real-space object images from a folder of .tif files.

For every .tif, ``m`` random N-by-N crops are taken and each crop is rotated
by a random multiple of 90 degrees. The crops are saved raw (``save_objects``)
and turned into complex objects for ``simulate.py`` by ``to_complex``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["load_image", "random_crops", "foreground_fraction", "prepare_objects", "save_objects", "load_objects", "to_complex"]


def load_image(path: str | Path) -> np.ndarray:
    """Read a single-channel image as a float32 (H, W) array.

    8-bit palette TIFFs (BBBC002) are read as their grayscale indices.
    """
    return np.asarray(Image.open(path), dtype=np.float32)


def random_crops(img: np.ndarray, m: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """``m`` random N-by-N crops of ``img``, each rotated by 0/90/180/270 degrees.

    Crops may overlap. Returns an (m, N, N) array.
    """
    H, W = img.shape
    if H < N or W < N:
        raise ValueError(f"image {img.shape} is smaller than crop size {N}")
    crops = np.empty((m, N, N), dtype=img.dtype)
    for k in range(m):
        i = rng.integers(0, H - N + 1)
        j = rng.integers(0, W - N + 1)
        crops[k] = np.rot90(img[i : i + N, j : j + N], k=rng.integers(0, 4))
    return crops


def foreground_fraction(crops: np.ndarray, margin: float = 10.0) -> np.ndarray:
    """Fraction of pixels in each crop that exceed the crop's median by more than ``margin``.

    The median is the background level (nuclei cover a small part of the field).
    """
    med = np.median(crops, axis=(-2, -1), keepdims=True)
    return (crops > med + margin).mean(axis=(-2, -1))


def prepare_objects(
    root: str | Path,
    m: int = 4,
    N: int = 200,
    seed: int | None = None,
    min_foreground: float = 0.005,
    margin: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Take ``m`` random N-by-N rotated crops from every .tif under ``root``.

    Crops where fewer than ``min_foreground`` of the pixels are more than
    ``margin`` gray levels above the crop's median are dropped as background-only
    (set ``min_foreground=0`` to keep everything).

    Returns ``(crops, source)``: crops is (n_kept, N, N) float32 and source[i] is
    the index of the file (in sorted order) crop i came from, so train/validation
    splits can be made by file.
    """
    files = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"no .tif files in {root}")
    rng = np.random.default_rng(seed)
    crops = np.concatenate([random_crops(load_image(f), m, N, rng) for f in files])
    source = np.repeat(np.arange(len(files)), m)
    keep = foreground_fraction(crops, margin) >= min_foreground
    return crops[keep], source[keep]


def save_objects(path: str | Path, crops: np.ndarray, source: np.ndarray, **meta) -> None:
    """Write crops, their source-file indices and any metadata (m, N, seed, ...) to a .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, crops=crops, source=source, **meta)


def load_objects(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(crops, source)`` back from a file written by ``save_objects``."""
    with np.load(path) as d:
        return d["crops"], d["source"]


def to_complex(
    crop: np.ndarray,
    phase_max: float = np.pi / 2,
    absorption: float = 0.3,
    percentiles: tuple[float, float] = (1.0, 99.5),
) -> np.ndarray:
    """Turn one real crop into a complex64 transmission function.

    The crop is stretched to [0, 1] between its ``percentiles``, then
    phase = phase_max * img and amplitude = 1 - absorption * img, so phase and
    absorption are correlated as in a real specimen.
    """
    lo, hi = np.percentile(crop, percentiles)
    hi = max(hi, lo + 1.0)  # guard against near-empty crops (1 gray level floor)
    img = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    return ((1.0 - absorption * img) * np.exp(1j * phase_max * img)).astype(np.complex64)
