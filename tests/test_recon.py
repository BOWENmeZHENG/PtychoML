import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cdtools.datasets import Ptycho2DDataset

from ptychoml.data.simulate import make_probe, simulate_cxi
from ptychoml.recon import batch, single
from ptychoml.recon.core import footprint_object, iter_reconstruction, load_ground_truth

WAVELENGTH = 1.24e-9
# Same tiny geometry as tests/test_simulate.py.
DETECTOR_GEOMETRY = {
    "distance": np.float32(0.112),
    "basis": np.array([[0.0, -1.2e-4], [-1.2e-4, 0.0], [0.0, 0.0]]),
    "corner": np.array([0.00592, 0.00618, 0.112]),
}
PROBE_SHAPE = (16, 16)
OBJ_SHAPE = (24, 24)
PADDING = 4
ITERATIONS = 3
# Options shared by the CLI tests, and the matching keyword arguments for iter_reconstruction.
CLI_ARGS = ["--obj-padding", str(PADDING), "--batch-size", "10"]
RECON_KWARGS = {"obj_padding": PADDING, "batch_size": 10}


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    """A folder in the processed-data layout holding one tiny simulated scan."""
    root = tmp_path_factory.mktemp("processed")
    torch.manual_seed(0)
    obj = torch.polar(1.0 - 0.3 * torch.rand(OBJ_SHAPE), torch.rand(OBJ_SHAPE))
    ref = SimpleNamespace(wavelength=WAVELENGTH, detector_geometry=DETECTOR_GEOMETRY)
    # Only the geometry read is stubbed; the simulated file is later read back with the real from_cxi.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Ptycho2DDataset, "from_cxi", lambda *args, **kwargs: ref)
        probe = make_probe(PROBE_SHAPE, 4.0)
        simulate_cxi(obj, root / "simulated" / "sim_00000.cxi", "geometry.cxi", probe=probe, step=4)
    return root


@pytest.fixture
def cxi(sim_dir):
    return sim_dir / "simulated" / "sim_00000.cxi"


class TestCore:
    def test_iter_reconstruction_yields_epochs_from_one(self, cxi):
        epochs = [e for e, _ in iter_reconstruction(cxi, iterations=ITERATIONS, **RECON_KWARGS)]
        assert epochs == [1, 2, 3]

    def test_footprint_object_has_ground_truth_shape(self, cxi):
        _, model = next(iter(iter_reconstruction(cxi, iterations=1, **RECON_KWARGS)))
        obj = footprint_object(model, PADDING)
        assert obj.shape == OBJ_SHAPE
        assert obj.dtype == np.complex64

    def test_load_ground_truth(self, cxi):
        assert load_ground_truth(cxi).shape == OBJ_SHAPE


class TestBatch:
    @staticmethod
    def run(sim_dir, out_dir, *extra):
        batch.main(
            [str(sim_dir), "--out-dir", str(out_dir), "--iterations", str(ITERATIONS), "--snapshot-at", "1"]
            + CLI_ARGS
            + list(extra)
        )

    def test_writes_snapshot_objects_and_meta(self, sim_dir, tmp_path):
        self.run(sim_dir, tmp_path)
        objs = np.load(tmp_path / "recon_00000.npy", mmap_mode="r")
        assert objs.shape == (2, *OBJ_SHAPE)
        assert objs.dtype == np.complex64
        assert json.loads((tmp_path / "meta.json").read_text())["iterations"] == [1, ITERATIONS]
        for epoch in (1, ITERATIONS):
            for name in ("amplitude", "phase"):
                assert (tmp_path / f"recon_00000_iter_{epoch}_obj_{name}.png").exists()
        assert not list(tmp_path.glob("*.tmp.npy"))

    def test_existing_outputs_are_skipped_unless_overwrite(self, sim_dir, tmp_path):
        out = tmp_path / "recon_00000.npy"
        self.run(sim_dir, tmp_path)
        out.write_bytes(b"keep")
        self.run(sim_dir, tmp_path)
        assert out.read_bytes() == b"keep"
        self.run(sim_dir, tmp_path, "--overwrite")
        assert out.read_bytes() != b"keep"

    def test_changed_settings_in_existing_folder_are_an_error(self, sim_dir, tmp_path):
        self.run(sim_dir, tmp_path)
        with pytest.raises(SystemExit):
            self.run(sim_dir, tmp_path, "--lr", "0.5")

    def test_snapshot_beyond_iterations_is_an_error(self, sim_dir, tmp_path):
        with pytest.raises(SystemExit):
            batch.main([str(sim_dir), "--out-dir", str(tmp_path), "--iterations", "2", "--snapshot-at", "5"])


class TestSingle:
    def test_saves_final_and_snapshot_results(self, cxi, tmp_path):
        out = tmp_path / "final.h5"
        single.main(
            ["--cxi", str(cxi), "--out", str(out), "--iterations", str(ITERATIONS), "--snapshot-at", "2"] + CLI_ARGS
        )
        names = {p.name for p in tmp_path.iterdir()}
        assert {"final.h5", "final_obj_amplitude.png", "final_obj_phase.png", "final_summary.png"} <= names
        assert {"sim_00000_standard_iter_2.h5", "sim_00000_standard_iter_2_obj_phase.png"} <= names
