"""Training loss.

    loss = 1.0 * BCE(mask)
         + 1.0 * [0.5 * per-instance Dice + 0.5 * image Dice]
         + 0.5 * Skeleton Recall (on the human-drawn spine tube)
         + 0.3 * BCE(spine head)

Why each term:
  * BCE keeps the probabilities calibrated pixel by pixel.
  * Image Dice alone lets one large filament own the gradient and trade a whole small filament for extra width on a
    big one; the metric is per instance with a hard IoU > 0.5 gate, so half of the Dice term is computed per
    ground-truth component, each inside its own margin-expanded box (neighbouring filaments excluded).
  * Skeleton Recall (Kirchhoff et al., ECCV 2024) puts uniform weight along the whole filament, including the faint
    ends where the mask evidence is weak. It uses MAGFiLO's drawn spine, not a morphological skeleton.
  * The spine head is deep supervision only; its map also feeds 5 of the filter's 15 features.
Pseudo-labelled frames (teacher-student) have no human spine: they skip Skeleton Recall, and their soft teacher
targets enter BCE, Dice and the spine head as probabilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TrainConfig


def image_dice_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    intersection = (probabilities * targets).sum((1, 2, 3))
    union = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()


def per_instance_dice_loss(probabilities: torch.Tensor, components: torch.Tensor, margin: int = 16) -> torch.Tensor:
    """Mean Dice over ground-truth connected components, each scored inside its bounding box grown by `margin`.
    Pixels of OTHER components inside the box are excluded rather than counted as background."""
    losses = []
    for index in range(probabilities.shape[0]):
        label_map, probability = components[index, 0], probabilities[index, 0]
        labels = torch.unique(label_map)
        for label in labels[labels > 0]:
            instance = label_map == label
            rows = torch.nonzero(instance.any(dim=1), as_tuple=False)
            cols = torch.nonzero(instance.any(dim=0), as_tuple=False)
            if not len(rows) or not len(cols):
                continue
            y0, y1 = max(int(rows[0]) - margin, 0), min(int(rows[-1]) + margin + 1, instance.shape[0])
            x0, x1 = max(int(cols[0]) - margin, 0), min(int(cols[-1]) + margin + 1, instance.shape[1])
            window = label_map[y0:y1, x0:x1]
            target = (window == label).float()
            valid = ((window == label) | (window == 0)).float()
            p = probability[y0:y1, x0:x1] * valid
            losses.append(1.0 - (2.0 * (p * target).sum() + 1.0) / (p.sum() + target.sum() + 1.0))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def skeleton_recall_loss(probabilities: torch.Tensor, spine: torch.Tensor) -> torch.Tensor:
    """1 - (probability mass recovered on the spine tube) / (tube size)."""
    covered = (probabilities * spine).sum((1, 2, 3))
    return (1.0 - covered / (spine.sum((1, 2, 3)) + 1.0)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, spine: torch.Tensor, components: torch.Tensor,
                      cfg: TrainConfig, pseudo: bool = False) -> torch.Tensor:
    """logits (B, 2, H, W): channel 0 = mask, 1 = spine. Reductions run in fp32 even under autocast."""
    logits, targets, spine = logits.float(), targets.float(), spine.float()
    mask_logits = logits[:, :1]
    probabilities = mask_logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
    mix = float(cfg.per_instance_dice_mix)
    dice = mix * per_instance_dice_loss(probabilities, components) + (1.0 - mix) * image_dice_loss(probabilities, targets)
    total = cfg.bce_weight * bce + cfg.dice_weight * dice
    if not pseudo:
        total = total + cfg.skeleton_recall_weight * skeleton_recall_loss(probabilities, spine)
    else:
        total = total + probabilities.sum() * 0.0
    return total + cfg.spine_aux_weight * F.binary_cross_entropy_with_logits(logits[:, 1:2], spine)
