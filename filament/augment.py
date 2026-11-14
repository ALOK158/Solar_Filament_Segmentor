"""Training-time augmentation of a (uint8 image, uint8/float target) pair at the network input size.

Order: optional rotation + scale (half of the samples in the v27/v31 recipes), the exact D4 group (rot90 + flips:
lossless for pixel labels), optional GONG-plausible photometrics, then gamma, blur, noise and a random CLAHE
strength. The target is warped with nearest-neighbour (bilinear-then-threshold would thin 2-3 px spine tubes on
diagonals); exposed borders take the off-disk sky level, not zero or a reflection, which at a zoom-out would
mirror a sliver of the limb into the corners.

`photometric_v2` and the affine are drawn only when enabled, so turning them off leaves every other random draw
unchanged.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import TrainConfig


def sky_level(image: np.ndarray) -> float:
    """Median of the four corner patches: a full-disk frame always leaves them off the Sun."""
    height, width = image.shape[:2]
    patch = max(8, min(height, width) // 32)
    corners = [image[:patch, :patch], image[:patch, -patch:], image[-patch:, :patch], image[-patch:, -patch:]]
    return float(np.median(np.concatenate([c.ravel() for c in corners])))


def random_affine(image: np.ndarray, target: np.ndarray, rng: np.random.Generator,
                  rotate_deg: float, scale_jitter: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation U(-r, r) degrees and scale U(1 - j, 1 + j) about the image centre."""
    height, width = image.shape[:2]
    angle = float(rng.uniform(-rotate_deg, rotate_deg))
    scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=sky_level(image))
    target = cv2.warpAffine(np.ascontiguousarray(target), matrix, (width, height), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, target


def random_dihedral(image: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One of the 8 rotations/reflections of the square."""
    k = int(rng.integers(4))
    image, target = np.rot90(image, k), np.rot90(target, k)
    if rng.random() < 0.5:
        image, target = image[:, ::-1], target[:, ::-1]
    if rng.random() < 0.5:
        image, target = image[::-1], target[::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], cells: int = 4) -> np.ndarray:
    """Zero-mean, unit-std noise varying over ~1/cells of the frame (bicubic-upsampled)."""
    coarse = rng.normal(0.0, 1.0, (cells, cells)).astype(np.float32)
    field = cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (field - field.mean()) / max(float(field.std()), 1e-6)


def photometric_v2(image: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
    """Each with probability `prob`: limb-darkening jitter (disk only), a low-frequency haze field, patchy seeing
    (blur mixed in through a smooth mask), JPEG recompression. None of them moves a pixel, so labels stay valid."""
    out = image.astype(np.float32)
    height, width = out.shape
    if rng.random() < prob:
        disk = image > 0.3 * float(np.percentile(image, 99))
        area = float(disk.sum())
        if area > 0.05 * height * width:
            ys, xs = np.nonzero(disk)
            cy, cx, radius = float(ys.mean()), float(xs.mean()), math.sqrt(area / math.pi)
            yy, xx = np.ogrid[:height, :width]
            r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / radius ** 2
            factor = 1.0 + float(rng.uniform(-0.15, 0.15)) * (np.minimum(r2, 1.0) - 0.5)
            out = np.where(disk, out * factor, out)
    if rng.random() < prob:
        out = out * (1.0 + float(rng.uniform(0.02, 0.08)) * _smooth_field(rng, (height, width)))
    if rng.random() < prob:
        blurred = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.8, 2.0)))
        alpha = np.clip(0.5 + 0.35 * _smooth_field(rng, (height, width)), 0.0, 1.0)
        out = alpha * blurred + (1.0 - alpha) * out
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < prob:
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return out


def photometric_base(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gamma U(0.7, 1.4); Gaussian blur (p 0.3, sigma 0.5-1.2); Gaussian noise (p 0.5, sigma 2-8 DN)."""
    gamma = float(rng.uniform(0.7, 1.4))
    image = np.clip(255.0 * (image / 255.0) ** gamma, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.5, 1.2)))
    if rng.random() < 0.5:
        noise = rng.normal(0, float(rng.uniform(2, 8)), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def augment(image: np.ndarray, target: np.ndarray, rng: np.random.Generator,
            cfg: TrainConfig) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns (augmented image, CLAHE clip limit to use, augmented target)."""
    if cfg.affine_prob > 0 and (cfg.rotate_deg > 0 or cfg.scale_jitter > 0) and rng.random() < cfg.affine_prob:
        image, target = random_affine(image, target, rng, cfg.rotate_deg, cfg.scale_jitter)
    image, target = random_dihedral(image, target, rng)
    if cfg.photometric_v2_prob > 0:
        image = photometric_v2(image, rng, cfg.photometric_v2_prob)
    image = photometric_base(image, rng)
    return image, float(rng.uniform(1.0, 4.0)), target
