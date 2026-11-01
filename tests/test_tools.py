import pytest
import torch

from ptychoml.cdtools_ext.tools import (
    ComplexUNet,
    ch2_to_complex,
    complex_to_2ch,
    process_obj,
)

H, W = 16, 16


@pytest.fixture
def complex_img():
    torch.manual_seed(0)
    return torch.complex(torch.randn(1, H, W), torch.randn(1, H, W))


@pytest.fixture
def unet():
    torch.manual_seed(0)
    return ComplexUNet(image_shape=(H, W), depth=2, base_channels=4)


class TestChannelConversion:
    def test_complex_to_2ch_shape_and_dtype(self, complex_img):
        out = complex_to_2ch(complex_img)
        assert out.shape == (1, 2, H, W)
        assert not out.is_complex()
        assert out.is_contiguous()

    def test_complex_to_2ch_channel_order(self, complex_img):
        out = complex_to_2ch(complex_img)
        assert torch.equal(out[:, 0], complex_img.real)
        assert torch.equal(out[:, 1], complex_img.imag)

    def test_ch2_to_complex_channel_order(self):
        x = torch.randn(1, 2, H, W)
        out = ch2_to_complex(x)
        assert torch.equal(out.real, x[:, 0])
        assert torch.equal(out.imag, x[:, 1])

    def test_roundtrip_complex(self, complex_img):
        assert torch.equal(ch2_to_complex(complex_to_2ch(complex_img)), complex_img)


class TestComplexUNet:
    def test_forward_preserves_shape_and_complex_dtype(self, unet, complex_img):
        out = unet(complex_img)
        assert out.shape == complex_img.shape
        assert out.is_complex()

    def test_backward_populates_grads(self, unet, complex_img):
        loss = unet(complex_img).abs().sum()
        loss.backward()
        grads = [p.grad for p in unet.parameters() if p.requires_grad]
        assert grads
        assert all(g is not None for g in grads)


class TestProcessObj:
    def test_3d_input_passes_through_model(self, unet, complex_img):
        unet.eval()
        with torch.no_grad():
            out = process_obj(complex_img, unet)
            expected = unet(complex_img)
        assert out.shape == (1, H, W)
        assert torch.equal(out, expected)

    def test_2d_input_returns_2d(self, unet):
        obj = torch.complex(torch.randn(H, W), torch.randn(H, W))
        out = process_obj(obj, unet)
        assert out.shape == (H, W)
