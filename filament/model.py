"""The segmentation network and its weight files.

FilamentUNet = a timm ConvNeXt-Tiny encoder (DINOv3-distilled weights, features at strides 4/8/16/32) + a U-Net
decoder + a small full-resolution stem. Two output channels: filament mask and filament spine (centre line).

Shapes for a 1536 x 1536 input:
    encoder   96 @ 384^2, 192 @ 192^2, 384 @ 96^2, 768 @ 48^2   (each GroupNorm-ed: DINOv3 stage 3 carries |x| ~ 131)
    decoder   768+384 -> 256 @ 96^2,  256+192 -> 128 @ 192^2,  128+96 -> 64 @ 384^2
    stem      3 -> 16 @ 1536^2 (two 3x3 convs on the raw input: the only path that sees 1-2 px filament edges)
    final     64 (upsampled x4) + 16 -> 32 @ 1536^2,  head 1x1 conv -> 2 logits (mask, spine)
31.9 M parameters. GroupNorm everywhere: training runs at batch size 1, where BatchNorm statistics are unreliable.

Weight files store fp16 tensors (61 MB per member; fp32 would exceed GitHub's 100 MB file limit) and are loaded
back to fp32, the precision the models run in.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_NAME = "convnext_tiny.dinov3_lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HEADS = ("mask", "spine")


class ConvBlock(nn.Sequential):
    """(3x3 conv -> GroupNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = math.gcd(8, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class FilamentUNet(nn.Module):
    """U-Net decoder over an encoder that returns a list of feature maps (shallow -> deep), plus a full-resolution
    stem of the input concatenated before the head."""

    def __init__(self, encoder: nn.Module, encoder_channels: Sequence[int],
                 decoder_channels: Sequence[int] = (256, 128, 64, 32), heads: int = len(HEADS),
                 input_channels: int = 3) -> None:
        super().__init__()
        self.encoder = encoder
        encoder_channels = list(encoder_channels)
        self.feature_norms = nn.ModuleList(nn.GroupNorm(math.gcd(32, c), c) for c in encoder_channels)
        skips = encoder_channels[:-1][::-1]
        if len(decoder_channels) != len(skips) + 1:
            raise ValueError(f"need {len(skips) + 1} decoder widths for {len(encoder_channels)} encoder levels")
        self.blocks = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for skip_channels, out_channels in zip(skips, decoder_channels[:-1]):
            self.blocks.append(ConvBlock(in_channels + skip_channels, out_channels))
            in_channels = out_channels
        self.stem = ConvBlock(input_channels, 16)
        self.final = ConvBlock(in_channels + 16, decoder_channels[-1])
        self.head = nn.Conv2d(decoder_channels[-1], heads, 1)
        # Channels beyond the 3 image channels (the refiner's probability / mask inputs, all in [0, 1]) get 0.5 / 0.25.
        extra = input_channels - 3
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN + (0.5,) * extra).view(1, input_channels, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD + (0.25,) * extra).view(1, input_channels, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) channels in [0, 1] -> (B, heads, H, W) logits. The encoder always runs in fp32, also under AMP:
        DINOv3 activations overflow fp16 (the first training run went NaN at epoch 4)."""
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.encoder(((images - self.mean) / self.std).float())
            features = [norm(f) for norm, f in zip(self.feature_norms, features)]
        x = features[-1]
        for block, skip in zip(self.blocks, features[:-1][::-1]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.final(torch.cat([x, self.stem(images)], dim=1)))


def build_model(encoder_name: str = ENCODER_NAME, pretrained: bool = False,
                decoder_channels: Sequence[int] = (256, 128, 64, 32), grad_checkpointing: bool = False,
                input_channels: int = 3, heads: int = len(HEADS)) -> FilamentUNet:
    """A FilamentUNet. `pretrained=True` downloads the DINOv3 encoder weights from the Hugging Face hub (training
    from scratch only); loading a shipped member needs no download. `input_channels` > 3 adapts the encoder's first
    convolution the timm way (pretrained RGB filters repeated and rescaled)."""
    import timm

    encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True, in_chans=input_channels)
    if grad_checkpointing:
        encoder.set_grad_checkpointing(True)
    return FilamentUNet(encoder, encoder.feature_info.channels(), decoder_channels, heads, input_channels)


def count_parameters(model: nn.Module) -> dict[str, int]:
    parts = {"encoder": model.encoder, "decoder": nn.ModuleList([model.feature_norms, model.blocks, model.final]),
             "stem": model.stem, "head": model.head}
    counts = {name: sum(p.numel() for p in part.parameters()) for name, part in parts.items()}
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


# ----------------------------------------------------------------------------------------------------------------
# Weight files
# ----------------------------------------------------------------------------------------------------------------

def _stored_tensors(model_or_state) -> dict[str, torch.Tensor]:
    """Learned tensors only: the fixed ImageNet mean/std buffers are rebuilt by the constructor."""
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return {k: v for k, v in state.items() if k not in ("mean", "std")}


def weights_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """sha256 over the fp16 tensor bytes in key order: identifies a member independently of the file container."""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().to("cpu", torch.float16).contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def save_member(model_or_state, path: str | Path, meta: dict | None = None) -> str:
    """Write a member as fp16 tensors plus a metadata dict. Returns the weights fingerprint."""
    state = {k: v.detach().to("cpu", torch.float16 if v.is_floating_point() else v.dtype).contiguous()
             for k, v in _stored_tensors(model_or_state).items()}
    fingerprint = weights_fingerprint(state)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "filament-unet/fp16/v1", "state_dict": state,
                "meta": {**(meta or {}), "fingerprint": fingerprint}}, path)
    return fingerprint


def read_member(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """(fp16 state dict, metadata) of a weight file. Also accepts a bare fp32 state dict (training checkpoints)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("format", "").startswith("filament-unet/"):
        return blob["state_dict"], blob.get("meta", {})
    return _stored_tensors(blob), {}


def load_member(path: str | Path, device: torch.device | str = "cpu") -> FilamentUNet:
    """A ready-to-run fp32 model in eval mode."""
    state, _ = read_member(path)
    model = build_model(pretrained=False)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if set(missing) - {"mean", "std"} or unexpected:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    return model.to(device).eval()


def load_members(weights_dir: str | Path, names: Sequence[str], device: torch.device | str = "cpu") -> list[FilamentUNet]:
    return [load_member(Path(weights_dir) / f"{name}.fp16.pt", device) for name in names]
