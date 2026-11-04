
"""Simulate ptychography datasets from real-space images.

Each input image is used as the complex object of a cdtools ``FancyPtycho``
model. ``FancyPtycho.sim_to_dataset`` runs the forward model
(interaction -> far-field propagation -> intensity) at the requested scan
positions and returns a ``Ptycho2DDataset``, which is written out as a .cxi.

Geometry (wavelength, detector, pixel size) is taken from an existing .cxi,
e.g. data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi.
"""

from pathlib import Path

import h5py
import numpy as np
import torch as t
from cdtools import tools
from cdtools.tools import data as cdtdata
from cdtools.datasets import Ptycho2DDataset
from cdtools.models import FancyPtycho

__all__ = ["grid_scan", "make_probe", "simulate_dataset", "simulate_cxi"]


def grid_scan(
    obj_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    step: int,
) -> t.Tensor:
    """Raster grid of probe-window corners (in object pixels) covering the object.

    Returns an (N, 2) float tensor of (i, j) positions.
    """
    max_i = obj_shape[0] - probe_shape[0]
    max_j = obj_shape[1] - probe_shape[1]
    if max_i < 0 or max_j < 0:
        raise ValueError(f"object {obj_shape} is smaller than probe {probe_shape}")
    ii, jj = np.meshgrid(np.arange(0, max_i + 1, step), np.arange(0, max_j + 1, step), indexing="ij")
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.float32)
    return t.as_tensor(pos)


def make_probe(shape: tuple[int, int] = (128, 128), radius: float = 20.0, defocus_phase: float = 0.0) -> t.Tensor:
    """Smooth circular-aperture probe (complex64), with optional quadratic phase."""
    ii, jj = np.meshgrid(np.arange(shape[0]) - shape[0] / 2, np.arange(shape[1]) - shape[1] / 2, indexing="ij")
    r2 = ii**2 + jj**2
    amp = np.exp(-((r2 / radius**2) ** 2))
    phase = defocus_phase * r2 / radius**2
    return t.as_tensor(amp * np.exp(1j * phase), dtype=t.complex64)


def _obj_basis(detector_geometry: dict, det_shape: tuple[int, int], wavelength: float) -> t.Tensor:
    """Object-plane pixel basis implied by the detector geometry."""
    return tools.initializers.exit_wave_geometry(
        t.as_tensor(detector_geometry["basis"], dtype=t.float32),
        det_shape,
        wavelength,
        float(detector_geometry["distance"]),
    )


def simulate_dataset(
    obj: t.Tensor,
    probe: t.Tensor,
    pix_positions: t.Tensor,
    wavelength: float,
    detector_geometry: dict,
    calculation_width: int | None = None,
) -> Ptycho2DDataset:
    """Simulate diffraction patterns of ``obj`` illuminated by ``probe``.

    obj: complex (H, W) object transmission function.
    probe: complex (h, w) probe; (h, w) is also the detector shape.
    pix_positions: (N, 2) probe-window corners in object pixels (i, j).
    detector_geometry: dict with 'distance' and 'basis' (as from cdtools).
    """
    probe = t.as_tensor(probe, dtype=t.complex64)
    obj = t.as_tensor(obj, dtype=t.complex64)
    pix_positions = t.as_tensor(pix_positions, dtype=t.float32)
    if probe.dim() == 2:
        probe = probe[None]  # (modes, h, w)
    det_shape = tuple(probe.shape[-2:])

    obj_basis = _obj_basis(detector_geometry, det_shape, wavelength)

    # pixel -> metres with the same basis the model uses to go back to pixels,
    # so min_translation=0 maps these positions straight onto the object array.
    translations = tools.interactions.pixel_to_translations(obj_basis, pix_positions)

    model = FancyPtycho(
        wavelength,
        detector_geometry,
        obj_basis,
        probe,
        obj,
        background=1e-6 * t.ones(det_shape, dtype=t.float32),
        panel_plot_mode=False,
    )
    indices = t.arange(len(pix_positions))
    with t.no_grad():
        return model.sim_to_dataset((indices, translations), calculation_width=calculation_width)


def _add_obj_and_probe(cxi_file: h5py.File, obj: t.Tensor, probe: t.Tensor, obj_basis: t.Tensor) -> None:
    """Store the ground-truth object and probe in an open .cxi file."""

    def to_np(x, dtype):
        return t.as_tensor(x).detach().cpu().numpy().astype(dtype)

    entry = cxi_file["entry_1"]
    image = entry.create_group("image_1")
    image["data"] = to_np(obj, np.complex64)
    image["obj_basis"] = to_np(obj_basis, np.float32)
    source = entry.require_group("instrument_1/source_1")
    source["probe"] = to_np(probe, np.complex64)


def simulate_cxi(
    obj: t.Tensor,
    out_path: str | Path,
    geometry_cxi: str | Path,
    probe: t.Tensor | None = None,
    step: int = 4,
    **kwargs,
) -> Ptycho2DDataset:
    """Simulate a scan of ``obj`` and save it to ``out_path`` (.cxi).

    Wavelength and detector geometry are read from ``geometry_cxi``. The ground
    truth ``obj`` and ``probe`` are stored alongside the diffraction data:
    ``/entry_1/image_1/data`` (obj), ``/entry_1/image_1/obj_basis`` (object
    pixel basis, metres) and ``/entry_1/instrument_1/source_1/probe`` (probe).
    """
    ref = Ptycho2DDataset.from_cxi(str(geometry_cxi), load_patterns=False)
    if probe is None:
        probe = make_probe()
    pos = grid_scan(tuple(obj.shape[-2:]), tuple(probe.shape[-2:]), step)
    dataset = simulate_dataset(obj, probe, pos, ref.wavelength, ref.detector_geometry, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj_basis = _obj_basis(ref.detector_geometry, tuple(probe.shape[-2:]), ref.wavelength)
    with cdtdata.create_cxi(str(out_path)) as f:
        dataset.to_cxi(f)
        _add_obj_and_probe(f, obj, probe, obj_basis)
    return dataset
