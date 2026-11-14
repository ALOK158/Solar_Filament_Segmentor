"""Ensemble + test-time augmentation -> native-resolution probability maps.

Each member sees the 1536 px input under each TTA view; sigmoid outputs are un-flipped and averaged over members x
views (all at the probability level), then resized back to the native 2048 px grid. A "model" is any callable
(1, 3, H, W) float32 tensor -> (1, 2, H, W) logits, so an ONNX Runtime session (onnx_backend.OrtSegmenter) drops
in for a PyTorch module unchanged.
"""
from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np
import torch

from .config import InferenceConfig, InputConfig
from .preprocess import network_input

Model = Callable[[torch.Tensor], torch.Tensor]


def _view_probability(model: Model, tensor: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """sigmoid(model(flip(x))), flipped back. Shape (1, heads, H, W)."""
    out = model(tensor.flip(dims) if dims else tensor).float().sigmoid()
    return out.flip(dims) if dims else out


@torch.no_grad()
def predict_small(models: Sequence[Model], channels: np.ndarray, device: torch.device | str,
                  cfg: InferenceConfig = InferenceConfig()) -> np.ndarray:
    """(3, S, S) network input -> (heads, S, S) mean probability over members x TTA views."""
    tensor = torch.from_numpy(channels)[None].to(device)
    total, count = None, 0
    for model in models:
        for dims in cfg.tta_flips:
            probability = _view_probability(model, tensor, dims)[0]
            total = probability.clone() if total is None else total + probability
            count += 1
    return (total / count).cpu().numpy()


def to_native(small: np.ndarray, height: int, width: int) -> np.ndarray:
    """(heads, S, S) -> (heads, height, width), bilinear, channel by channel."""
    return np.stack([cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR) for c in small])


def predict_maps(models: Sequence[Model], image: np.ndarray, device: torch.device | str,
                 input_cfg: InputConfig = InputConfig(), cfg: InferenceConfig = InferenceConfig()) -> np.ndarray:
    """uint8 (H, W) image -> float32 (2, H, W): [filament probability, spine probability] at native resolution."""
    small = predict_small(models, network_input(image, input_cfg), device, cfg)
    return to_native(small, *image.shape)


def quantise(maps: np.ndarray) -> np.ndarray:
    """Probability maps -> uint8 (p * 255, rounded): how validation maps are stored (4 MB per 2048^2 channel)."""
    return np.round(maps * 255).astype(np.uint8)


def dequantise(maps: np.ndarray) -> np.ndarray:
    return maps.astype(np.float32) / 255.0
