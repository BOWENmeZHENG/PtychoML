""" 
Input: real-space images, different shapes

A few probes: 128 by 128

scan position: grid within the image

physics-based simulation: diffraction pattern generation 
based on the input images and scan positions   

forward model: applies the physics-based simulation to generate diffraction patterns

cdtools forward model location: 
.venv/lib/python3.12/site-packages/cdtools/models/base.py
def forward(self, *args):
    return self.measurement(self.forward_propagator(self.interaction(*args)))

.venv/lib/python3.12/site-packages/cdtools/models/fancy_ptycho.py
FancyPtycho needs: wavelength, detector_geometry, obj_basis, probe_guess, obj_guess

Try getting them from data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi
can from_dataset work??
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(str(CXI))

Use: .venv/lib/python3.12/site-packages/cdtools/tools/data/data.py
get_wavelength, get_detector_geometry, 

wavelength: 1.2400e-09

detector_geometry new corner [0.00768, 0.00768, 0.112]
{'distance': np.float32(0.112), 'basis': array([[ 0.     , -0.00012],
       [-0.00012,  0.     ],
       [ 0.     ,  0.     ]]), 'corner': array([0.00592, 0.00618, 0.112  ])}

obj_basis for a 128 by 128 probe, use 9.0415e-09
tensor([[ 0.0000e+00, -1.1128e-08],
        [-1.1128e-08,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00]])

bring own probe_guess and obj_guess

Actually, can I use sim_to_dataset to generate??


Output: .cxi file for each each image


"""


"""Simulate ptychography datasets from real-space images.

Each input image is used as the complex object of a cdtools ``FancyPtycho``
model. ``FancyPtycho.sim_to_dataset`` runs the forward model
(interaction -> far-field propagation -> intensity) at the requested scan
positions and returns a ``Ptycho2DDataset``, which is written out as a .cxi.

Geometry (wavelength, detector, pixel size) is taken from an existing .cxi,
e.g. data/toy/AuBalls_700ms_30nmStep_3_6SS_filter.cxi.
"""

from pathlib import Path

import numpy as np
import torch as t
from cdtools import tools
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

    obj_basis = tools.initializers.exit_wave_geometry(
        t.as_tensor(detector_geometry["basis"], dtype=t.float32),
        det_shape,
        wavelength,
        float(detector_geometry["distance"]),
    )

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


def simulate_cxi(
    obj: t.Tensor,
    out_path: str | Path,
    geometry_cxi: str | Path,
    probe: t.Tensor | None = None,
    step: int = 32,
    **kwargs,
) -> Ptycho2DDataset:
    """Simulate a scan of ``obj`` and save it to ``out_path`` (.cxi).

    Wavelength and detector geometry are read from ``geometry_cxi``.
    """
    ref = Ptycho2DDataset.from_cxi(str(geometry_cxi), load_patterns=False)
    if probe is None:
        probe = make_probe()
    pos = grid_scan(tuple(obj.shape[-2:]), tuple(probe.shape[-2:]), step)
    dataset = simulate_dataset(obj, probe, pos, ref.wavelength, ref.detector_geometry, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_cxi(str(out_path))
    return dataset
