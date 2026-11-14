"""Train a U-Net to map an early reconstruction (real and imaginary channels) to a later one, tracked with MLflow.

The data is the output folder of ``ptychoml.recon.batch`` (see ``ptychoml.train.data``). The network is the
``ComplexUNet`` of ``ptychoml.cdtools_ext``, trained on its two-channel ``net``. Defaults come from the ``train``
section of ``params.yaml``, which DVC tracks, and every one can be overridden on the command line.

Each run logs the parameters, the loss and PSNR of every epoch, the same numbers for the identity baseline (the input
used as the prediction, which the network has to beat), a figure of validation examples and the checkpoint. The
checkpoint is also written to ``--out`` and the final numbers to ``--metrics``. Browse the runs with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from ptychoml.cdtools_ext.tools import ComplexUNet, ch2_to_complex, save_checkpoint
from ptychoml.train.data import load_training_data

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[3]
PARAMS_PATH = ROOT_DIR / "params.yaml"
DVC_LOCK_PATH = ROOT_DIR / "dvc.lock"
OUT_PATH = ROOT_DIR / "models/tunet.pt"
METRICS_PATH = ROOT_DIR / "metrics/train.json"
TRACKING_URI = "sqlite:///mlflow.db"  # relative to the working directory; MLFLOW_TRACKING_URI overrides it
EXPERIMENT = "ptychoml"
PANEL_SAMPLES = 3  # validation examples in the logged figure

# The `train` section of params.yaml: name -> (type, help). All of them are command-line options too.
PARAMS = {
    "input_iter": (int, "Iteration of the input object"),
    "target_iter": (int, "Iteration of the target object"),
    "val_fraction": (float, "Fraction of the source images held out for validation"),
    "seed": (int, "Random seed for the split, the weights and the batch order"),
    "epochs": (int, "Maximum number of epochs"),
    "batch_size": (int, "Batch size"),
    "lr": (float, "Adam learning rate"),
    "patience": (int, "Stop when the validation loss has not improved for this many epochs"),
    "depth": (int, "U-Net depth"),
    "base_channels": (int, "Channels of the first U-Net level"),
    "growth_rate": (int, "Channel growth factor per U-Net level"),
    "hidden_rate": (int, "Channels between the two convolutions of a level, relative to the level's channels"),
}


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Load the ``train`` section of params.yaml, which DVC tracks for the train stage."""
    with open(path) as f:
        return yaml.safe_load(f)["train"]


def psnr(mse: float, peak: float) -> float:
    """Peak signal-to-noise ratio in dB; ``peak`` is the largest absolute value of a real or imaginary channel."""
    return float("inf") if mse == 0 else 10 * math.log10(peak**2 / mse)


def run_epoch(
    model: ComplexUNet, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None = None
) -> float:
    """One pass over ``loader``: train if an ``optimizer`` is given, else evaluate. Returns the mean MSE per value."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model.net(x), y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


@torch.no_grad()
def predict(model: ComplexUNet, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Network output for the (N, 2, H, W) batch ``x``, on the CPU."""
    model.eval()
    return model.net(x.to(device)).cpu()


def validation_figure(x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, scale: float) -> Figure:
    """Amplitude and phase of input, prediction, target and their difference, two rows per sample."""
    x, pred, y = (ch2_to_complex(t).numpy() * scale for t in (x, pred, y))
    n = len(x)
    fig = Figure(figsize=(10, 5 * n), layout="constrained")
    axes = fig.subplots(2 * n, 4, squeeze=False)
    for i in range(n):
        images = {
            "amplitude": ([np.abs(x[i]), np.abs(pred[i]), np.abs(y[i]), np.abs(pred[i] - y[i])], "viridis", "magma"),
            "phase": ([np.angle(a) for a in (x[i], pred[i], y[i], pred[i] * np.conj(y[i]))], "twilight", "twilight"),
        }
        for row, (name, (imgs, cmap, err_cmap)) in enumerate(images.items()):
            vmin, vmax = (
                (-np.pi, np.pi) if name == "phase" else (min(map(np.min, imgs[:3])), max(map(np.max, imgs[:3])))
            )
            for col, (img, title) in enumerate(zip(imgs, ("input", "prediction", "target", "difference"), strict=True)):
                ax = axes[2 * i + row, col]
                if col < 3:
                    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                elif name == "phase":
                    ax.imshow(img, cmap=err_cmap, vmin=-np.pi, vmax=np.pi)
                else:
                    ax.imshow(img, cmap=err_cmap)
                ax.set_title(f"{title} {name}", fontsize=8)
                ax.axis("off")
    return fig


def dvc_hash(path: Path) -> str | None:
    """MD5 that ``dvc.lock`` records for the output ``path``, which identifies the exact data, or ``None``."""
    try:
        relative = path.resolve().relative_to(ROOT_DIR).as_posix()
        for stage in yaml.safe_load(DVC_LOCK_PATH.read_text())["stages"].values():
            for out in stage.get("outs", []):
                if out["path"] == relative:
                    return out["md5"]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        pass
    return None


def train(
    recon_dir: Path,
    params: dict,
    out: Path = OUT_PATH,
    metrics_path: Path = METRICS_PATH,
    device: str | None = None,
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT,
    processed_dir: Path | None = None,
) -> dict:
    """Train on the pairs in ``recon_dir`` and track the run with MLflow.

    Args:
        recon_dir: Output folder of ``ptychoml.recon.batch``.
        params: Values for every key of ``PARAMS``.
        out: Where to write the checkpoint of the best epoch (see ``ptychoml.cdtools_ext.tools.save_checkpoint``).
        metrics_path: Where to write the final numbers as JSON.
        device: Torch device, e.g. 'cuda' or 'mps' (default: CPU).
        tracking_uri: MLflow tracking URI (default: ``MLFLOW_TRACKING_URI`` if set, else a local SQLite file).
        experiment: MLflow experiment name.
        processed_dir: Folder with ``objects.npz`` (default: the one named in the recon folder's ``meta.json``).

    Returns:
        The final numbers that are written to ``metrics_path``.
    """
    device = torch.device(device or "cpu")
    torch.manual_seed(params["seed"])
    data = load_training_data(
        recon_dir,
        params["input_iter"],
        params["target_iter"],
        val_fraction=params["val_fraction"],
        seed=params["seed"],
        processed_dir=processed_dir,
    )
    model = ComplexUNet(
        image_shape=data.image_shape,
        depth=params["depth"],
        base_channels=params["base_channels"],
        growth_rate=params["growth_rate"],
        hidden_rate=params["hidden_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    shuffle = torch.Generator().manual_seed(params["seed"])
    train_loader = DataLoader(data.train, batch_size=params["batch_size"], shuffle=True, generator=shuffle)
    val_loader = DataLoader(data.val, batch_size=params["batch_size"])

    val_x, val_y = data.val.tensors
    peak = val_y.abs().max().item()
    baseline_loss = F.mse_loss(val_x, val_y).item()  # the identity map: use the input as the prediction

    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                **params,
                "recon_dir": Path(recon_dir).name,
                "device": str(device),
                "scale": data.scale,
                "n_train": len(data.train),
                "n_val": len(data.val),
                "n_parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        if data_hash := dvc_hash(Path(recon_dir)):
            mlflow.set_tag("dvc.data_md5", data_hash)
        mlflow.log_metrics({"baseline_val_loss": baseline_loss, "baseline_val_psnr": psnr(baseline_loss, peak)})

        best_loss, best_epoch, best_state = math.inf, 0, None
        for epoch in range(1, params["epochs"] + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_psnr": psnr(val_loss, peak)}, step=epoch
            )
            logger.info("epoch %d: train %.5f, val %.5f (identity %.5f)", epoch, train_loss, val_loss, baseline_loss)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= params["patience"]:
                logger.info("Stopping: no improvement since epoch %d", best_epoch)
                break

        model.load_state_dict(best_state)
        results = {
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "val_loss": best_loss,
            "val_psnr": psnr(best_loss, peak),
            "baseline_val_loss": baseline_loss,
            "baseline_val_psnr": psnr(baseline_loss, peak),
        }
        mlflow.log_metrics({f"best_{k}" if k.startswith("val") else k: v for k, v in results.items()})

        n = min(PANEL_SAMPLES, len(data.val))
        fig = validation_figure(val_x[:n], predict(model, val_x[:n], device), val_y[:n], data.scale)
        mlflow.log_figure(fig, "validation_examples.png")

        model.cpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out, model, data.scale)
        mlflow.log_artifact(str(out))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = load_params()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recon_dir", type=Path, help="Output folder of ptychoml.recon.batch")
    for name, (kind, help_) in PARAMS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=kind,
            default=defaults[name],
            help=f"{help_} (default: %(default)s)",
        )
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Checkpoint file (default: %(default)s)")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH, help="Final metrics JSON (default: %(default)s)")
    parser.add_argument("--device", default=None, help="Torch device, e.g. 'cuda' or 'mps' (default: CPU)")
    parser.add_argument("--tracking-uri", default=None, help=f"MLflow tracking URI (default: {TRACKING_URI})")
    parser.add_argument("--experiment", default=EXPERIMENT, help="MLflow experiment name (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the training; argv defaults to ``sys.argv[1:]`` when ``None``."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = train(
        args.recon_dir,
        {name: getattr(args, name) for name in PARAMS},
        out=args.out,
        metrics_path=args.metrics,
        device=args.device,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
    )
    logger.info("Done: %s", results)


if __name__ == "__main__":
    main()
