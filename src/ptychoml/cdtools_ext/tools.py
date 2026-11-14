from pathlib import Path

import torch
from dlsia.core.networks import tunet
from torch import nn

__all__ = [
    "ComplexUNet",
    "ch2_to_complex",
    "complex_to_2ch",
    "load_checkpoint",
    "process_obj",
    "save_checkpoint",
]


class ComplexUNet(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int],
        depth: int,
        base_channels: int,
        growth_rate: int = 2,
        hidden_rate: int = 1,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.hidden_rate = hidden_rate
        self.net = tunet.TUNet(
            image_shape=self.image_shape,
            in_channels=2,
            out_channels=2,
            depth=self.depth,
            base_channels=self.base_channels,
            growth_rate=self.growth_rate,
            hidden_rate=self.hidden_rate,
        )

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        x_2ch = complex_to_2ch(x_complex)  # [1,2,H,W]
        y_2ch = self.net(x_2ch)  # [1,2,H,W]
        y_cplx = ch2_to_complex(y_2ch)  # [1,H,W] complex
        return y_cplx


def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    x: complex [1,H,W]
    returns: real [1, 2, H, W]
    """
    xr = torch.view_as_real(x)  # [1,H,W,2]
    xr = xr.permute(0, 3, 1, 2)  # [1,2,H,W]
    return xr.contiguous()


def ch2_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    x: real [1, 2, H, W]
    returns: complex [1, H, W]
    """
    x = x.permute(0, 2, 3, 1).contiguous()  # [1,H,W,2]
    return torch.view_as_complex(x)


def save_checkpoint(path: str | Path, model: ComplexUNet, scale: float) -> None:
    """Save the weights, the arguments needed to rebuild ``model`` and the ``scale`` its data was divided by."""
    config = {
        "image_shape": tuple(model.image_shape),
        "depth": model.depth,
        "base_channels": model.base_channels,
        "growth_rate": model.growth_rate,
        "hidden_rate": model.hidden_rate,
    }
    torch.save({"config": config, "state_dict": model.state_dict(), "scale": scale}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[ComplexUNet, float]:
    """Rebuild a ``ComplexUNet`` saved by ``save_checkpoint``, in eval mode, and return it with its data scale."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = ComplexUNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["scale"]


def process_obj(obj, model):
    was_2d = obj.dim() == 2
    if was_2d:
        obj = obj.unsqueeze(0)  # Shape: (1, H_obj, W_obj)
    obj_processed = model(obj)
    if was_2d:
        return obj_processed.squeeze(0)
    return obj_processed
