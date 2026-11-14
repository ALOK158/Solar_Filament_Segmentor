"""Grayscale H-alpha image -> the three network input channels.

All three channels are deterministic image transforms, computed from the single test image only (no metadata, no
neighbouring frames), identically in training and inference:
    0. background-normalised intensity: removes limb darkening and vignetting
    1. CLAHE: local contrast
    2. Hessian dark-ridge response: thin dark lines, integrated along their length
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import InputConfig


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Whole image -> size x size with area interpolation (averages, so faint filaments dim but survive)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Binary target(s) (H, W) or (H, W, C) -> size x size: area interpolation, then >= 0.5.

    A pixel is foreground if most of it was. ('> 0' would widen every target by up to a pixel per side, which caps
    IoU on thin filaments near the 0.5 match threshold.) The thinnest MAGFiLO filament is 1.86 px wide, so at the
    0.75 scale every column keeps a majority pixel."""
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (resized >= 0.5).astype(mask.dtype)


def background_normalised(image: np.ndarray, sigma: float = 64.0, gain: float = 0.5) -> np.ndarray:
    """image / local background * gain, clipped to [0, 1].

    Filament visibility is contrast against the LOCAL chromosphere, not absolute brightness. Dividing by a heavily
    blurred copy removes limb darkening and the vignetting gradient in one step."""
    img = image.astype(np.float32)
    background = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img / np.maximum(background, 1e-3) * gain, 0.0, 1.0)


def clahe(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, as float32 in [0, 1]."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(image).astype(np.float32) / 255.0


def ridge_response(image: np.ndarray, scales=(2.0, 4.0, 8.0), percentile: float = 99.5) -> np.ndarray:
    """Multi-scale Hessian ridge filter for DARK lines, normalised to [0, 1].

    A dark ridge has a positive second derivative across its axis, so the larger Hessian eigenvalue is kept.
    Each scale is multiplied by sigma^2 so the scales are comparable, the maximum over scales is taken, and the
    result is divided by its `percentile`-th value."""
    img = image.astype(np.float32) / 255.0
    best = np.zeros_like(img)
    for sigma in scales:
        smooth = cv2.GaussianBlur(img, (0, 0), float(sigma))
        gxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        spread = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy ** 2, 0.0))
        larger = 0.5 * ((gxx + gyy) + spread)
        best = np.maximum(best, (sigma ** 2) * larger)
    high = float(np.percentile(best, percentile))
    return np.clip(best / max(high, 1e-6), 0.0, 1.0)


def to_channels(image: np.ndarray, cfg: InputConfig = InputConfig(), clahe_clip: float | None = None) -> np.ndarray:
    """uint8 (H, W) -> float32 (3, H, W). `clahe_clip` overrides the config (training jitters it)."""
    return np.stack([background_normalised(image, cfg.background_sigma, cfg.background_gain),
                     clahe(image, cfg.clahe_clip if clahe_clip is None else clahe_clip, cfg.clahe_grid),
                     ridge_response(image, cfg.ridge_scales, cfg.ridge_percentile)])


def network_input(image: np.ndarray, cfg: InputConfig = InputConfig()) -> np.ndarray:
    """Native uint8 image -> the (3, input_size, input_size) float32 array the network sees at inference."""
    return to_channels(resize_image(image, cfg.input_size), cfg)
