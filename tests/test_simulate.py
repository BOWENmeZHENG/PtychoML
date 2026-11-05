from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from cdtools.datasets import Ptycho2DDataset

from ptychoml.data.simulate import (
    grid_scan,
    make_probe,
    simulate_cxi,
    simulate_cxi_batch,
    simulate_dataset,
)

WAVELENGTH = 1.24e-9
# Same geometry as data/toy/AuBalls_*.cxi, which is DVC-tracked and so absent in CI.
DETECTOR_GEOMETRY = {
    "distance": np.float32(0.112),
    "basis": np.array([[0.0, -1.2e-4], [-1.2e-4, 0.0], [0.0, 0.0]]),
    "corner": np.array([0.00592, 0.00618, 0.112]),
}
PROBE_SHAPE = (16, 16)
OBJ_SHAPE = (24, 24)


@pytest.fixture
def probe():
    return make_probe(PROBE_SHAPE, radius=4.0)


@pytest.fixture
def obj():
    torch.manual_seed(0)
    amp = 1.0 - 0.3 * torch.rand(OBJ_SHAPE)
    return torch.polar(amp, torch.rand(OBJ_SHAPE))


@pytest.fixture
def geometry_cxi(monkeypatch):
    """Stand in for reading the geometry from a real .cxi."""
    ref = SimpleNamespace(wavelength=WAVELENGTH, detector_geometry=DETECTOR_GEOMETRY)
    monkeypatch.setattr(Ptycho2DDataset, "from_cxi", lambda *args, **kwargs: ref)
    return "geometry.cxi"


class TestGridScan:
    def test_covers_object_with_step(self):
        pos = grid_scan((10, 12), (4, 4), step=2)
        # i in 0..6 step 2 -> 4 values, j in 0..8 step 2 -> 5 values
        assert pos.shape == (20, 2)
        assert pos.dtype == torch.float32
        assert pos[:, 0].max() == 6
        assert pos[:, 1].max() == 8

    def test_row_major_order(self):
        pos = grid_scan((6, 6), (4, 4), step=1)
        assert pos[:3].tolist() == [[0, 0], [0, 1], [0, 2]]
        assert pos[3].tolist() == [1, 0]

    def test_positions_keep_probe_window_inside_object(self):
        obj_shape, probe_shape = (30, 25), (8, 6)
        pos = grid_scan(obj_shape, probe_shape, step=3)
        assert (pos[:, 0] + probe_shape[0] <= obj_shape[0]).all()
        assert (pos[:, 1] + probe_shape[1] <= obj_shape[1]).all()

    def test_probe_same_size_as_object_gives_single_point(self):
        assert grid_scan((8, 8), (8, 8), step=4).tolist() == [[0, 0]]

    @pytest.mark.parametrize(("obj_shape", "probe_shape"), [((7, 10), (8, 8)), ((10, 7), (8, 8))])
    def test_probe_larger_than_object_raises(self, obj_shape, probe_shape):
        with pytest.raises(ValueError, match="smaller than probe"):
            grid_scan(obj_shape, probe_shape, step=1)


class TestSimulateDataset:
    def test_one_pattern_per_position(self, obj, probe):
        pos = grid_scan(OBJ_SHAPE, PROBE_SHAPE, step=4)
        ds = simulate_dataset(obj, probe, pos, WAVELENGTH, DETECTOR_GEOMETRY)
        assert isinstance(ds, Ptycho2DDataset)
        assert len(ds) == len(pos)
        assert ds.patterns.shape == (len(pos), *PROBE_SHAPE)
        assert ds.wavelength == pytest.approx(WAVELENGTH)

    def test_patterns_are_finite_and_nonnegative(self, obj, probe):
        pos = grid_scan(OBJ_SHAPE, PROBE_SHAPE, step=8)
        ds = simulate_dataset(obj, probe, pos, WAVELENGTH, DETECTOR_GEOMETRY)
        assert torch.isfinite(ds.patterns).all()
        assert (ds.patterns >= 0).all()
        assert ds.patterns.sum() > 0

    def test_accepts_numpy_inputs(self, obj, probe):
        pos = grid_scan(OBJ_SHAPE, PROBE_SHAPE, step=8)
        ds = simulate_dataset(obj.numpy(), probe.numpy(), pos.numpy(), WAVELENGTH, DETECTOR_GEOMETRY)
        assert len(ds) == len(pos)



class TestSimulateCxi:
    def test_writes_diffraction_data_and_ground_truth(self, obj, probe, geometry_cxi, tmp_path):
        out = tmp_path / "nested" / "scan.cxi"
        ds = simulate_cxi(obj, out, geometry_cxi, probe=probe, step=8)
        assert out.exists()
        with h5py.File(out, "r") as f:
            assert f["entry_1/image_1/data"].shape == OBJ_SHAPE
            assert f["entry_1/image_1/data"].dtype == np.complex64
            assert np.allclose(f["entry_1/image_1/data"][()], obj.numpy())
            assert f["entry_1/image_1/obj_basis"].shape == (3, 2)
            assert f["entry_1/instrument_1/source_1/probe"].shape == PROBE_SHAPE
            assert np.allclose(f["entry_1/instrument_1/source_1/probe"][()], probe.numpy())
            assert f["entry_1/data_1/data"].shape[0] == len(ds)

    def test_default_probe_is_used_when_omitted(self, geometry_cxi, tmp_path):
        big = torch.ones((128, 128), dtype=torch.complex64)
        out = tmp_path / "scan.cxi"
        simulate_cxi(big, out, geometry_cxi, step=64)
        with h5py.File(out, "r") as f:
            assert f["entry_1/instrument_1/source_1/probe"].shape == (128, 128)

    def test_object_smaller_than_probe_raises(self, probe, geometry_cxi, tmp_path):
        tiny = torch.ones((8, 8), dtype=torch.complex64)
        with pytest.raises(ValueError, match="smaller than probe"):
            simulate_cxi(tiny, tmp_path / "scan.cxi", geometry_cxi, probe=probe)


class TestSimulateCxiBatch:
    def test_writes_one_file_per_object(self, obj, probe, geometry_cxi, tmp_path):
        objects = torch.stack([obj, obj * 0.9])
        paths = simulate_cxi_batch(objects, tmp_path / "out", geometry_cxi, probe=probe, step=8)
        assert [p.name for p in paths] == ["sim_00000.cxi", "sim_00001.cxi"]
        assert all(p.exists() for p in paths)
        with h5py.File(paths[1], "r") as f:
            assert np.allclose(f["entry_1/image_1/data"][()], (obj * 0.9).numpy())

    def test_custom_prefix(self, obj, probe, geometry_cxi, tmp_path):
        paths = simulate_cxi_batch(obj[None], tmp_path, geometry_cxi, probe=probe, step=8, prefix="train")
        assert paths[0].name == "train_00000.cxi"

    def test_non_3d_input_raises(self, obj, probe, geometry_cxi, tmp_path):
        with pytest.raises(ValueError, match="num_objects, H, W"):
            simulate_cxi_batch(obj, tmp_path, geometry_cxi, probe=probe)
