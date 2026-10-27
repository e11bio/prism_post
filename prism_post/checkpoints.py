"""
Build each model, load the released weights, and wrap it for volara.

The released checkpoints are from before the current volara-ml. The model.pt
files on the bucket were saved from a private training package, so they can't
be loaded here. Instead we build each model from models.py (same definitions
as prism_training), load the model_checkpoint_* weights, and save the result
next to the checkpoint. That only needs to happen once. Newer models are saved
in a way TorchModel loads directly, so for those you'd only need a SPECS entry
pointing at the released file.

SPECS also holds what volara needs to tile the prediction: the smallest
input/output shape, the step size, number of output channels and output range.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from funlib.learn.torch.models import UNet as FunlibUNet
from volara_ml.models import TorchModel

from prism_post.config import NEIGHBORHOOD
from prism_post.models import (
    ChannelAgnosticModel,
    ChannelAgnosticWrapper,
    MtlsdModel,
    SigmoidWrapper,
    UNet,
)

# Models (should match the training scripts in prism_training)


def build_enhanced() -> torch.nn.Module:
    """
    Enhancement. Channel agnostic, so the same unet runs on each of the 18
    channels on its own. It predicts a residual that gets added back onto
    the input.
    """
    ds_fact = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
    num_levels = len(ds_fact) + 1
    return ChannelAgnosticModel(
        num_fmaps=12,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=[[(3, 3, 3), (3, 3, 3)]] * num_levels,
        kernel_size_up=[[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1),
        upsample_mode="trilinear",
        use_residual=True,
        iter_channels=True,
        add_residual=True,
    )


def build_affs() -> torch.nn.Module:
    """Affinities + lsds from the 18 enhanced channels."""
    ds_fact = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
    num_levels = len(ds_fact) + 1
    return MtlsdModel(
        in_channels=18,
        num_fmaps=12,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=[[(3, 3, 3), (3, 3, 3)]] * num_levels,
        kernel_size_up=[[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1),
        constant_upsample=True,
        num_affs=len(NEIGHBORHOOD),
        num_fmaps_out=18,
    )


def build_uniform() -> torch.nn.Module:
    """
    Uniform embedding. Just 1x1x1 convs with no downsampling, so really a
    small mlp per voxel, mapping the 18 channels to a 24d vector. Similar
    barcodes end up with a high cosine similarity.
    """
    fmaps = 1024
    unet = UNet(
        in_channels=18,
        num_fmaps=fmaps,
        fmap_inc_factor=2,
        downsample_factors=[],
        kernel_size_down=[[(1, 1, 1)] * 6],
    )
    return torch.nn.Sequential(unet, torch.nn.Conv3d(fmaps, 24, kernel_size=1))


class _KeepChannelDim(torch.nn.Module):
    """``(b, z, y, x) -> (b, 1, z, y, x)`` so the output has a channel axis."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y.unsqueeze(1) if y.dim() == x.dim() - 1 else y


def build_binary() -> torch.nn.Module:
    """
    Barcode expression. Channel agnostic, runs the unet on each channel,
    takes the max over channels and a sigmoid, giving a single channel
    probability that any barcode is expressed.
    """
    fmaps = 12
    unet = FunlibUNet(
        in_channels=1,
        num_fmaps=fmaps,
        fmap_inc_factor=2,
        downsample_factors=[(1, 2, 2), (2, 2, 2), (2, 2, 2)],
        kernel_size_down=[
            [(1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
        ],
        kernel_size_up=[
            [(1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
        ],
        constant_upsample=True,
    )
    return SigmoidWrapper(
        ChannelAgnosticWrapper(
            torch.nn.Sequential(unet, torch.nn.Conv3d(fmaps, 1, kernel_size=1))
        )
    )


# model specs


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[], torch.nn.Module]
    checkpoint: str
    min_input_shape: tuple[int, int, int]
    min_output_shape: tuple[int, int, int]
    min_step_shape: tuple[int, int, int]
    out_channels: list[int] | int | None
    out_range: tuple[float, float]
    pred_size_growth: tuple[int, int, int]
    wrap: Callable[[torch.nn.Module], torch.nn.Module] | None = None


SPECS: dict[str, ModelSpec] = {
    "enhanced": ModelSpec(
        build=build_enhanced,
        checkpoint="model_checkpoint_60000",
        min_input_shape=(52, 148, 148),
        min_output_shape=(20, 56, 56),
        min_step_shape=(2, 8, 8),
        out_channels=None,  # same as input: 18
        out_range=(0, 1),
        pred_size_growth=(40, 80, 80),
    ),
    "affs": ModelSpec(
        build=build_affs,
        checkpoint="model_checkpoint_140000",
        min_input_shape=(36, 100, 100),
        min_output_shape=(4, 8, 8),
        min_step_shape=(2, 8, 8),
        out_channels=[len(NEIGHBORHOOD), 10],  # affs, lsds
        out_range=(0, 1),
        pred_size_growth=(96, 256, 256),
    ),
    "uniform": ModelSpec(
        build=build_uniform,
        checkpoint="model_checkpoint_10000",
        min_input_shape=(1, 212, 212),
        min_output_shape=(1, 212, 212),
        min_step_shape=(1, 1, 1),
        out_channels=24,
        out_range=(-1, 1),  # embeddings are regularised towards unit norm
        pred_size_growth=(19, 0, 0),
    ),
    "binary_channel_agnostic": ModelSpec(
        build=build_binary,
        checkpoint="model_checkpoint_100000",
        min_input_shape=(48, 148, 148),
        min_output_shape=(16, 56, 56),
        min_step_shape=(4, 8, 8),
        out_channels=1,
        out_range=(0, 1),
        pred_size_growth=(128, 256, 256),
        wrap=_KeepChannelDim,
    ),
}

SAVED_MODEL = "prism_post_model.pt"


def load_weights(name: str, checkpoint_dir: Path) -> torch.nn.Module:
    """Build a model and load its weights (on cpu)."""
    spec = SPECS[name]
    model = spec.build()
    checkpoint = torch.load(
        checkpoint_dir / name / spec.checkpoint, map_location="cpu", weights_only=True
    )
    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    if spec.wrap is not None:
        model = spec.wrap(model)
    return model.eval()


def torch_model(name: str, checkpoint_dir: Path, in_channels: int = 18) -> TorchModel:
    """
    Wrap a model for volara. The first time this is called for a model it
    saves the built model + weights to <checkpoint_dir>/<name>/prism_post_model.pt
    """
    spec = SPECS[name]
    save_path = checkpoint_dir / name / SAVED_MODEL
    if not save_path.exists():
        print(f"building {name} and saving to {save_path}")
        torch.save(load_weights(name, checkpoint_dir), save_path)

    return TorchModel(
        save_path=save_path,
        checkpoint_file=None,  # weights are already in the saved model
        in_channels=in_channels,
        out_channels=spec.out_channels,
        min_input_shape=spec.min_input_shape,
        min_output_shape=spec.min_output_shape,
        min_step_shape=spec.min_step_shape,
        out_range=spec.out_range,
        pred_size_growth=spec.pred_size_growth,
    )
