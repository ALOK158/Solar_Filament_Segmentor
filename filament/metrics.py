"""Panoptic quality and the rubric's agreement statistics.

The competition PQ scores a prediction against EVERY annotator record of an image and pools the counts over all
(image, record) pairs:  PQ = sum(IoU of matches) / (TP + FP/2 + FN/2),  a match being IoU > 0.5 under Hungarian
assignment. SQ = mean IoU of the matches, RQ = TP / (TP + FP/2 + FN/2), PQ = SQ x RQ.

The rubric (organizers' self-evaluation notebook) adds distributions over ALL overlapping (ground truth,
prediction) pairs: IoU, Dice, and the overlap degree, i.e. 1:0 misses, 1:n fragmented filaments, 0:1 false
alarms and n:1 merged ones.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from .data import AnnotationIndex, ground_truth_sets

ROW_COLUMNS = ["filename", "annotator", "pq", "sq", "rq", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred"]


def pairwise_iou(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix via pycocotools RLEs."""
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    gt = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in gt_masks]
    pred = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in pred_masks]
    return np.asarray(mask_utils.iou(gt, pred, [0] * len(pred)), dtype=np.float64)


def panoptic_quality(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray],
                     match_iou: float = 0.5) -> dict:
    """PQ of one prediction against one annotator record."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
    matched = np.zeros(0)
    if n_gt and n_pred:
        iou = pairwise_iou(gt_masks, pred_masks)
        rows, cols = linear_sum_assignment(-iou)
        matched = iou[rows, cols]
        matched = matched[matched > match_iou]
    tp = int(len(matched))
    fp, fn = n_pred - tp, n_gt - tp
    sum_iou = float(matched.sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": sum_iou / denominator if denominator else 1.0, "sq": sum_iou / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 1.0, "tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou}


def score_image(filename: str, pred_masks: Sequence[np.ndarray], index: AnnotationIndex,
                match_iou: float = 0.5, gt_sets: list | None = None) -> list[dict]:
    """One row per annotator record of the image, ready for aggregate_pq."""
    gt_sets = gt_sets if gt_sets is not None else ground_truth_sets(index, filename)
    return [{"filename": filename, "annotator": a, **panoptic_quality(gt, pred_masks, match_iou),
             "n_gt": len(gt), "n_pred": len(pred_masks)} for a, gt in enumerate(gt_sets)]


def aggregate_pq(rows: pd.DataFrame | Iterable[dict]) -> dict:
    """Pooled PQ / SQ / RQ over score_image rows: counts are summed over all (image, record) rows, never averaged."""
    rows = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=ROW_COLUMNS)
    if len(rows) == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0, "n_images": 0}
    tp, fp, fn = int(rows["tp"].sum()), int(rows["fp"].sum()), int(rows["fn"].sum())
    sum_iou = float(rows["sum_iou"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": sum_iou / denominator if denominator else 1.0, "sq": sum_iou / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 1.0, "tp": tp, "fp": fp, "fn": fn,
            "n_images": int(rows["filename"].nunique())}


def bootstrap_pq(rows: pd.DataFrame, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of pooled PQ, resampling IMAGES (an image's records stay together)."""
    rng = np.random.default_rng(seed)
    per_image = rows.groupby("filename")[["sum_iou", "tp", "fp", "fn"]].sum().to_numpy()
    draws = []
    for _ in range(n):
        s = per_image[rng.integers(0, len(per_image), len(per_image))].sum(axis=0)
        draws.append(s[0] / max(s[1] + 0.5 * s[2] + 0.5 * s[3], 1e-9))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_ratio(per_image: pd.DataFrame, numerator: str, denominator: str, n: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """95% interval of sum(numerator) / sum(denominator), resampling images (e.g. missed / ground-truth count)."""
    rng = np.random.default_rng(seed)
    num, den = per_image[numerator].to_numpy(float), per_image[denominator].to_numpy(float)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(num), len(num))
        draws.append(num[pick].sum() / max(den[pick].sum(), 1e-9))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def human_agreement_rows(index: AnnotationIndex, filenames: Iterable[str], match_iou: float = 0.5) -> pd.DataFrame:
    """Each annotator's drawing scored as a prediction against every OTHER annotator of the same image (images
    with one record are skipped): the human-vs-human reference for every number we report."""
    rows = []
    for filename in filenames:
        sets = ground_truth_sets(index, filename)
        for p, predicted in enumerate(sets):
            for a, truth in enumerate(sets):
                if a != p:
                    rows.append({"filename": filename, "annotator": a, "predictor": p,
                                 **panoptic_quality(truth, predicted, match_iou),
                                 "n_gt": len(truth), "n_pred": len(predicted)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Rubric: IoU / Dice over all overlapping pairs, and overlap degrees
# ----------------------------------------------------------------------------------------------------------------

class AgreementStats:
    """Accumulates the rubric statistics over many (ground truth, prediction) comparisons."""

    def __init__(self) -> None:
        self.iou, self.dice, self.gt_degree, self.pred_degree = [], [], [], []
        self.tp, self.fp, self.fn, self.tp_iou = 0, 0, 0, 0.0

    def add(self, gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> None:
        if not gt_masks or not pred_masks:
            self.gt_degree += [0] * len(gt_masks)
            self.pred_degree += [0] * len(pred_masks)
            self.fn += len(gt_masks)
            self.fp += len(pred_masks)
            return
        iou = pairwise_iou(gt_masks, pred_masks)
        dice = 2 * iou / (1 + iou)                         # Dice = 2I / (A + B) = 2 IoU / (1 + IoU)
        touching = iou > 0
        self.iou += iou[touching].tolist()
        self.dice += dice[touching].tolist()
        self.gt_degree += touching.sum(axis=1).tolist()
        self.pred_degree += touching.sum(axis=0).tolist()
        hit = iou > 0.5                                    # disjoint predictions: at most one hit per GT
        self.tp += int(hit.sum())
        self.tp_iou += float(iou[hit].sum())
        self.fp += int((hit.sum(axis=0) == 0).sum())
        self.fn += int((hit.sum(axis=1) == 0).sum())

    def summary(self) -> dict:
        iou, dice = np.array(self.iou), np.array(self.dice)
        gt, pred = Counter(self.gt_degree), Counter(self.pred_degree)
        n_gt, n_pred = max(len(self.gt_degree), 1), max(len(self.pred_degree), 1)
        denominator = max(self.tp + 0.5 * self.fp + 0.5 * self.fn, 1e-9)
        return {"PQ": self.tp_iou / denominator, "SQ": self.tp_iou / max(self.tp, 1), "RQ": self.tp / denominator,
                "IoU (pair mean)": float(iou.mean()) if iou.size else 0.0,
                "Dice (pair mean)": float(dice.mean()) if dice.size else 0.0,
                "missed 1:0": gt.get(0, 0) / n_gt, "1:1 (GT side)": gt.get(1, 0) / n_gt,
                "fragmented 1:n": sum(v for k, v in gt.items() if k >= 2) / n_gt,
                "false alarm 0:1": pred.get(0, 0) / n_pred, "1:1 (prediction side)": pred.get(1, 0) / n_pred,
                "merged n:1": sum(v for k, v in pred.items() if k >= 2) / n_pred,
                "ground-truth filaments": len(self.gt_degree), "predicted filaments": len(self.pred_degree)}


def thin_pixel_share(masks: Sequence[np.ndarray], radius: int = 2) -> float:
    """Share of mask pixels a morphological opening of radius r removes (structures narrower than 2r+1 px):
    a proxy for barbs and filament tips."""
    thin, total = thin_pixel_counts(masks, radius)
    return thin / max(total, 1)


def thin_pixel_counts(masks: Sequence[np.ndarray], radius: int = 2) -> tuple[int, int]:
    """(thin pixels, all pixels) over `masks`; sum these per image to get a dataset-level share."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    thin = total = 0
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, mask.shape[0])
        x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, mask.shape[1])
        sub = mask[y0:y1, x0:x1].astype(np.uint8)
        opened = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel).astype(bool)
        thin += int((sub.astype(bool) & ~opened).sum())
        total += int(sub.sum())
    return thin, total
