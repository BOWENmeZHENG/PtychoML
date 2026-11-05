import numpy as np
import pytest
from PIL import Image

from ptychoml.data.objects import (
    load_objects,
    prepare_objects,
    random_crops,
    save_objects,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def image_dir(tmp_path):
    """Two 32x32 uint8 .tif images with a bright blob on a flat background, plus a non-image file."""
    for name, level in [("a.tif", 50), ("b.tiff", 80)]:
        arr = np.full((32, 32), level, dtype=np.uint8)
        arr[8:20, 8:20] = 200
        Image.fromarray(arr).save(tmp_path / name)
    (tmp_path / "notes.txt").write_text("ignore me")
    return tmp_path


class TestRandomCrops:
    def test_shape_and_dtype(self, rng):
        img = rng.random((40, 50)).astype(np.float32)
        crops = random_crops(img, m=5, N=16, rng=rng)
        assert crops.shape == (5, 16, 16)
        assert crops.dtype == img.dtype

    @pytest.mark.parametrize("shape", [(7, 20), (20, 7)])
    def test_image_smaller_than_crop_raises(self, rng, shape):
        with pytest.raises(ValueError, match="smaller than crop size"):
            random_crops(np.zeros(shape, dtype=np.float32), m=1, N=8, rng=rng)


class TestPrepareObjects:
    def test_shapes_and_source_indices(self, image_dir):
        objects, source = prepare_objects(image_dir, m=3, N=16, seed=0, min_foreground=0.0)
        assert objects.shape == (6, 16, 16)
        assert objects.dtype == np.complex64
        assert source.tolist() == [0, 0, 0, 1, 1, 1]

    def test_seed_is_reproducible(self, image_dir):
        a, _ = prepare_objects(image_dir, m=3, N=16, seed=5)
        b, _ = prepare_objects(image_dir, m=3, N=16, seed=5)
        assert np.array_equal(a, b)

    def test_drops_background_only_crops(self, tmp_path):
        Image.fromarray(np.full((32, 32), 50, dtype=np.uint8)).save(tmp_path / "flat.tif")
        busy = np.full((32, 32), 200, dtype=np.uint8)
        busy[::2] = 50  # every crop is half bright pixels
        Image.fromarray(busy).save(tmp_path / "busy.tif")
        objects, source = prepare_objects(tmp_path, m=4, N=16, seed=0, min_foreground=0.1)
        assert len(objects) == len(source) == 4
        assert set(source.tolist()) == {0}  # files sort as busy.tif (0), flat.tif (1); flat is dropped entirely

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no .tif files"):
            prepare_objects(tmp_path)


class TestSaveLoadObjects:
    def test_roundtrip(self, tmp_path, rng):
        crops = (rng.random((3, 8, 8)) + 1j * rng.random((3, 8, 8))).astype(np.complex64)
        source = np.array([0, 0, 1])
        path = tmp_path / "nested" / "objects.npz"
        save_objects(path, crops, source, m=2, N=8, seed=1)
        loaded_crops, loaded_source = load_objects(path)
        assert np.array_equal(loaded_crops, crops)
        assert np.array_equal(loaded_source, source)
