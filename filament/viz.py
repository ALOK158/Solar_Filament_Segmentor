"""Figures for the notebook (matplotlib). One palette everywhere; identity is never carried by colour alone
(every multi-series figure has a legend, boxes carry text labels)."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from .decode import bounding_box

# Categorical slots in fixed order (validated for colour-vision deficiency as adjacent pairs).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KEPT, DROPPED, MISSED = COLORS[0], COLORS[1], COLORS[6]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    """Recessive axes and grid, thin lines."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=COLORS),
    })


def _image_axis(ax, image: np.ndarray, title: str = "", window=None) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255 if image.dtype == np.uint8 else None)
    ax.set_title(title)
    ax.axis("off")
    if window is not None:
        (x0, y0, x1, y1) = window
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)


def _outline(mask: np.ndarray) -> np.ndarray:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_instances(ax, instances: Sequence[np.ndarray], color: str, labels: Sequence[str] | None = None,
                   dashed: bool = False, fill: bool = True, box: bool = True) -> None:
    """Filled outline + bounding box + text label per instance."""
    for k, mask in enumerate(instances):
        for contour in _outline(mask):
            pts = contour[:, 0, :]
            if fill:
                ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.35, linewidth=0)
            ax.plot(np.r_[pts[:, 0], pts[:1, 0]], np.r_[pts[:, 1], pts[:1, 1]], color=color, linewidth=0.8)
        if box:
            x0, y0, x1, y1 = bounding_box(mask)
            ax.add_patch(patches.Rectangle((x0 - 4, y0 - 4), x1 - x0 + 8, y1 - y0 + 8, fill=False, edgecolor=color,
                                           linewidth=1.0, linestyle="--" if dashed else "-"))
            if labels is not None:
                ax.text(x0 - 4, y0 - 10, labels[k], color="white", fontsize=6.5,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor=color, edgecolor="none", alpha=0.9))


def show_channels(image: np.ndarray, channels: np.ndarray, title: str = "") -> plt.Figure:
    """The raw frame next to the three network input channels."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    _image_axis(axes[0], image, "input frame (2048 px)")
    for ax, channel, name in zip(axes[1:], channels, ["background-normalised", "CLAHE", "dark-ridge response"]):
        ax.imshow(channel, cmap="gray")
        ax.set_title(f"{name} ({channel.shape[0]} px)")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def segmentation_panel(image: np.ndarray, probability: np.ndarray, kept: Sequence[np.ndarray],
                       kept_scores: Sequence[float], dropped: Sequence[np.ndarray], dropped_scores: Sequence[float],
                       title: str = "", window=None) -> plt.Figure:
    """Frame | filament probability | final instances (solid boxes, filter score) and filtered-out candidates
    (dashed boxes)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    _image_axis(axes[0], image, "H-alpha frame", window)
    axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("ensemble filament probability")
    axes[1].axis("off")
    if window is not None:
        axes[1].set_xlim(window[0], window[2]); axes[1].set_ylim(window[3], window[1])
    _image_axis(axes[2], image, f"{len(kept)} filaments kept, {len(dropped)} dropped by the filter", window)
    draw_instances(axes[2], kept, KEPT, [f"{k + 1}: {s:.2f}" for k, s in enumerate(kept_scores)])
    draw_instances(axes[2], dropped, DROPPED, [f"x {s:.2f}" for s in dropped_scores], dashed=True, fill=False)
    handles = [patches.Patch(color=KEPT, label="kept (label: id, filter score)"),
               patches.Patch(edgecolor=DROPPED, facecolor="none", linestyle="--", label="dropped by the filter")]
    axes[2].legend(handles=handles, loc="lower left", fontsize=7, frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def instance_gallery(items: Sequence[dict], columns: int = 3) -> plt.Figure:
    """Grid of frames with their final instances and boxes. items: {image, instances, scores, title}."""
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for ax, item in zip(axes.ravel(), items):
        _image_axis(ax, item["image"], item.get("title", ""))
        draw_instances(ax, item["instances"], KEPT, [f"{k + 1}: {s:.2f}" for k, s in enumerate(item["scores"])])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def validation_panel(image: np.ndarray, gt_sets: Sequence[Sequence[np.ndarray]], predicted: Sequence[np.ndarray],
                     iou_by_record: Sequence[np.ndarray], title: str = "", window=None) -> plt.Figure:
    """Each annotator's drawing (left panels) against the prediction (right): matched predictions solid, false
    alarms dashed, missed filaments outlined. A prediction counts as matched if it matches ANY annotator."""
    n = len(gt_sets)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5.2))
    for a, (ax, gt) in enumerate(zip(axes[:-1], gt_sets)):
        iou = iou_by_record[a]
        hit = (iou > 0.5).any(axis=1) if iou.size else np.zeros(len(gt), bool)
        _image_axis(ax, image, f"annotator {a + 1}: {len(gt)} filaments, {int(hit.sum())} found", window)
        draw_instances(ax, [m for m, h in zip(gt, hit) if h], COLORS[2], box=False)
        draw_instances(ax, [m for m, h in zip(gt, hit) if not h], MISSED, fill=False, dashed=True)
    matched = np.zeros(len(predicted), bool)
    for iou in iou_by_record:
        if iou.size:
            matched |= (iou > 0.5).any(axis=0)
    best = np.max([iou.max(axis=0) for iou in iou_by_record if iou.size], axis=0) if predicted and any(
        i.size for i in iou_by_record) else np.zeros(len(predicted))
    ax = axes[-1]
    _image_axis(ax, image, f"prediction: {int(matched.sum())} matched, {int((~matched).sum())} unmatched", window)
    draw_instances(ax, [m for m, h in zip(predicted, matched) if h], KEPT,
                   [f"IoU {b:.2f}" for b, h in zip(best, matched) if h])
    draw_instances(ax, [m for m, h in zip(predicted, matched) if not h], DROPPED,
                   [f"IoU {b:.2f}" for b, h in zip(best, matched) if not h], dashed=True, fill=False)
    handles = [patches.Patch(color=COLORS[2], label="annotated, found"),
               patches.Patch(edgecolor=MISSED, facecolor="none", linestyle="--", label="annotated, missed"),
               patches.Patch(color=KEPT, label="predicted, matched (IoU > 0.5)"),
               patches.Patch(edgecolor=DROPPED, facecolor="none", linestyle="--", label="predicted, unmatched")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def refinement_panel(image: np.ndarray, gt_sets: Sequence[Sequence[np.ndarray]], seeds: Sequence[np.ndarray],
                     refined: Sequence[np.ndarray], title: str = "", window=None) -> plt.Figure:
    """Stage 2 on one image: every annotator's drawing, then the kept instances before (seeds) and after the
    refiner, each labelled with its best IoU against any annotator."""
    from .metrics import pairwise_iou

    def best_iou(masks):
        scores = [pairwise_iou(list(gt), list(masks)).max(axis=0) for gt in gt_sets if gt and masks]
        return np.max(scores, axis=0) if scores else np.zeros(len(masks))

    n = len(gt_sets)
    fig, axes = plt.subplots(1, n + 2, figsize=(4.2 * (n + 2), 4.6))
    for a, (ax, gt) in enumerate(zip(axes[:n], gt_sets)):
        _image_axis(ax, image, f"annotator {a + 1}: {len(gt)} filaments", window)
        draw_instances(ax, gt, COLORS[2], box=False)
    for ax, masks, name, color in ((axes[n], seeds, "before (stage 1)", DROPPED), (axes[n + 1], refined, "after (refiner)", KEPT)):
        scores = best_iou(masks)
        _image_axis(ax, image, f"{name}: mean best IoU {scores.mean():.2f}" if len(scores) else name, window)
        draw_instances(ax, masks, color, [f"{s:.2f}" for s in scores])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def learning_curves(histories: Mapping[str, pd.DataFrame], title: str = "") -> plt.Figure:
    """Training loss and validation PQ per epoch, one line per run (two panels, never two y-axes).
    Validation PQ here is in-training: one view, no filter, so it sits below the deployed numbers."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    for (name, history), color in zip(histories.items(), COLORS):
        left.plot(history["epoch"], history["train_loss"], color=color, label=name)
        val = history.dropna(subset=["val_pq"]) if "val_pq" in history else history.iloc[0:0]
        if len(val):
            right.plot(val["epoch"], val["val_pq"], color=color, marker="o", label=name)
            right.annotate(f"{val['val_pq'].iloc[-1]:.3f}", (val["epoch"].iloc[-1], val["val_pq"].iloc[-1]),
                           textcoords="offset points", xytext=(4, 0), fontsize=7, color=INK, va="center")
    left.set(title="training loss", xlabel="epoch", ylabel="loss (mean over the epoch)")
    right.set(title="validation PQ during training (1 view, no filter)", xlabel="epoch", ylabel="pooled PQ")
    for ax in (left, right):
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def filter_figure(oof: pd.DataFrame, curve: pd.DataFrame, closed_form: float, chosen: float) -> plt.Figure:
    """Left: out-of-fold filter scores of candidates no annotator drew vs candidates at least one drew.
    Right: pooled PQ against the drop threshold. `oof` has columns match and score."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(0, 1, 26)
    oof_scores = oof["score"].to_numpy()
    drawn = oof["match"].to_numpy() > 0
    left.hist(oof_scores[~drawn], bins=bins, color=DROPPED, alpha=0.75, label=f"no annotator drew it ({(~drawn).sum()})")
    left.hist(oof_scores[drawn], bins=bins, color=KEPT, alpha=0.6, label=f"at least one did ({drawn.sum()})")
    left.axvline(chosen, color=INK, linewidth=1.2, linestyle="--")
    left.text(chosen, left.get_ylim()[1] * 0.95, f" threshold {chosen:.3f}", fontsize=7, color=INK, va="top")
    left.set(title="filter score (out-of-fold) by outcome", xlabel="predicted share of annotators", ylabel="candidates")
    left.legend(fontsize=8)
    c = curve.sort_values("threshold")
    right.plot(c["threshold"], c["pq"], color=KEPT, marker="o", label="pooled PQ")
    right.axvline(closed_form, color=MUTED, linewidth=1, linestyle=":", label=f"closed form 0.5 PQ/SQ = {closed_form:.3f}")
    right.set(title="PQ vs drop threshold (fold 0)", xlabel="threshold", ylabel="pooled PQ")
    right.legend(fontsize=8)
    fig.tight_layout()
    return fig


def agreement_bars(ours: Mapping[str, float], human: Mapping[str, float], keys: Sequence[str]) -> plt.Figure:
    """The rubric's quantities, ours vs human-vs-human, side by side (one shared axis per quantity)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(2.1 * len(keys), 3.2))
    for ax, key in zip(np.atleast_1d(axes), keys):
        values = [ours[key], human[key]]
        bars = ax.bar(["ours", "human"], values, color=[KEPT, COLORS[2]], width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}" if v < 1 else f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7, color=INK)
        ax.set_title(key, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    fig.tight_layout()
    return fig


def pair_histograms(ours, human, name: str) -> plt.Figure:
    """Distribution of IoU or Dice over all overlapping (ground truth, prediction) pairs."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(ours, bins=bins, density=True, histtype="step", linewidth=2, color=KEPT, label=f"ours (n={len(ours)})")
    ax.hist(human, bins=bins, density=True, histtype="step", linewidth=2, color=COLORS[2], label=f"human vs human (n={len(human)})")
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set(title=f"{name} over all overlapping pairs", xlabel=name, ylabel="density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def timing_bars(table: pd.DataFrame, title: str = "") -> plt.Figure:
    """Seconds per image by pipeline stage."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    t = table.sort_values("s_per_image")
    ax.barh(t.index, t["s_per_image"], color=KEPT, height=0.6)
    for y, v in enumerate(t["s_per_image"]):
        ax.text(v, y, f" {v:.2f} s", va="center", fontsize=7, color=INK)
    ax.set(title=title or "time per image by stage", xlabel="seconds per image")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def data_overview(images: pd.DataFrame) -> plt.Figure:
    """Images per site, annotators per image, filaments per annotator."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    sites = images["site"].value_counts().sort_index()
    axes[0].bar(sites.index, sites.values, color=KEPT, width=0.6)
    axes[0].set(title="images per GONG site", ylabel="images")
    ann = images["annotators"].value_counts().sort_index()
    axes[1].bar(ann.index.astype(str), ann.values, color=KEPT, width=0.6)
    axes[1].set(title="annotators per image", xlabel="annotators", ylabel="images")
    axes[2].hist(images["filaments_mean"], bins=np.arange(0, images["filaments_mean"].max() + 2), color=KEPT)
    axes[2].set(title="filaments per annotator, per image", xlabel="filaments (mean over annotators)", ylabel="images")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
