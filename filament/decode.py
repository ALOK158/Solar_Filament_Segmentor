"""Probability map -> filament instances.

Threshold at 0.60, split into 8-connected components, drop components under 240 native pixels. Each surviving
component is one filament. Components are disjoint by construction, so stage 1 never produces overlapping
instances (the host rejects overlaps); the refiner's paste-back keeps them disjoint.

Rejected alternatives, all measured on fold 0: hysteresis thresholding, joining collinear fragments across gaps
(more merged filaments than humans draw), growing or shrinking instances (only loses PQ), a learned
instance-separation head, and a Mask2Former-style query decoder (its stride-4 masks cap PQ at 0.64 vs 0.74 here).
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import DecodeConfig


def decode_instances(probability: np.ndarray, cfg: DecodeConfig = DecodeConfig()) -> list[np.ndarray]:
    """float (H, W) probability -> list of boolean instance masks, in connected-component label order."""
    binary = (probability >= cfg.threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [labels == k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= cfg.min_area]


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), inclusive pixel coordinates."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def label_image(instances: list[np.ndarray], shape: tuple[int, int] | None = None) -> np.ndarray:
    """int32 map: 0 = background, k = instance k (1-based, in list order)."""
    shape = shape if shape is not None else instances[0].shape
    labels = np.zeros(shape, dtype=np.int32)
    for k, mask in enumerate(instances, start=1):
        labels[mask] = k
    return labels
