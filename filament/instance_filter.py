"""The learned keep/drop filter for decoded instances.

Each candidate filament gets 15 features computed from its own mask and the two probability maps (nothing else:
no file name, date or neighbouring frame). A gradient-boosted regressor predicts the fraction of annotators who
drew a matching filament (IoU > 0.5). Candidates scoring below a threshold are dropped.

Fitting (fold-0 validation maps only, never the training images the networks saw):
  1. Five regressors, cross-fitted with GroupKFold by observation DATE (same-day frames show the same filaments).
  2. Their out-of-fold scores pick the threshold: the closed form 0.5 * PQ / SQ, unless a nearby grid value beats
     it by more than 0.002 PQ.
  3. Deployment averages the five models; the threshold is moved to the quantile that reproduces the calibrated
     drop rate, because an average of models scores a little differently from out-of-fold predictions.
Measured on fold 0: PQ 0.422 -> 0.457, dropping 28% of the candidates (AUC 0.80).
"""
from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from .config import DecodeConfig, FilterConfig
from .data import AnnotationIndex, ground_truth_sets, observation_date
from .decode import decode_instances
from .metrics import ROW_COLUMNS, aggregate_pq, pairwise_iou, score_image

MASK_FEATURES = ["area", "length", "width", "elongation", "n_parts",
                 "prob_mean", "prob_max", "prob_p90", "contrast", "radius"]
SPINE_FEATURES = ["s_mean", "s_max", "s_p90", "s_rel", "s_cover"]
FEATURES = MASK_FEATURES + SPINE_FEATURES


def instance_features(probability: np.ndarray, mask: np.ndarray, spine: np.ndarray,
                      spine_max: float | None = None) -> dict[str, float]:
    """Shape, confidence and spine features of one instance (computed on its bounding box).

    `radius` is the distance of the centroid from the image centre in disk radii: limb filaments are drawn less
    consistently. The spine features ask "does a filament axis run through this blob", an axis independent of the
    mask probability; `s_rel` is relative to the image's spine peak because the spine head never saturates."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    y0, y1 = max(ys.min() - 3, 0), min(ys.max() + 4, height)
    x0, x1 = max(xs.min() - 3, 0), min(xs.max() + 4, width)
    sub, prob = mask[y0:y1, x0:x1].astype(np.uint8), probability[y0:y1, x0:x1]
    inside = prob[sub.astype(bool)]
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    coords -= coords.mean(axis=0)
    eig = np.sort(np.linalg.eigvalsh(coords.T @ coords / max(len(coords) - 1, 1)))[::-1]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    ring = cv2.dilate(sub, np.ones((5, 5), np.uint8)).astype(bool) & ~sub.astype(bool)
    inside_spine = spine[mask.astype(bool)]
    peak = float(spine_max if spine_max is not None else spine.max())
    return {
        "area": float(len(ys)),
        "length": float(4.0 * math.sqrt(max(eig[0], 0.0))),
        "width": float(2.0 * len(ys) / max(perimeter, 1.0)),        # area / half-perimeter: a ribbon's width
        "elongation": float(math.sqrt((eig[0] + 1e-6) / (eig[1] + 1e-6))),
        "n_parts": float(cv2.connectedComponents(sub, connectivity=8)[0] - 1),
        "prob_mean": float(inside.mean()),
        "prob_max": float(inside.max()),
        "prob_p90": float(np.percentile(inside, 90)),
        "contrast": float(inside.mean() - (prob[ring].mean() if ring.any() else 0.0)),
        "radius": float(math.hypot(ys.mean() - height / 2, xs.mean() - width / 2) / (min(height, width) / 2)),
        "s_mean": float(inside_spine.mean()),
        "s_max": float(inside_spine.max()),
        "s_p90": float(np.percentile(inside_spine, 90)),
        "s_rel": float(inside_spine.max() / max(peak, 1e-6)),
        "s_cover": float((inside_spine >= 0.5 * peak).mean()),
    }


def features_table(probability: np.ndarray, spine: np.ndarray, instances: Sequence[np.ndarray]) -> pd.DataFrame:
    spine_max = float(spine.max())
    return pd.DataFrame([instance_features(probability, m, spine, spine_max) for m in instances], columns=FEATURES)


@dataclass
class InstanceFilter:
    """Deployed filter: the averaged cross-fitted regressors and the drop threshold."""
    models: list
    threshold: float
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    report: dict = field(default_factory=dict)
    sklearn_version: str = ""
    # Out-of-fold scores of the validation candidates (filename, instance, match, score): the honest basis for
    # every filtered validation number, since the deployed average of the fold models has seen all of them.
    oof: pd.DataFrame | None = None

    def scores(self, probability: np.ndarray, spine: np.ndarray, instances: Sequence[np.ndarray]) -> np.ndarray:
        """Predicted share of annotators who would draw each instance, in [0, 1]."""
        if not instances:
            return np.zeros(0)
        return predict_scores(self.models, features_table(probability, spine, instances), self.features)

    def apply(self, probability: np.ndarray, spine: np.ndarray,
              instances: Sequence[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """(kept instances, their scores, scores of ALL candidates)."""
        scores = self.scores(probability, spine, instances)
        keep = scores >= self.threshold
        return [m for m, k in zip(instances, keep) if k], scores[keep], scores

    def save(self, path: str | Path) -> Path:
        import sklearn

        self.sklearn_version = sklearn.__version__
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=4)
        return Path(path)

    @staticmethod
    def load(path: str | Path) -> "InstanceFilter":
        import sklearn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # sklearn's version-mismatch warning: checked below instead
            with open(path, "rb") as handle:
                loaded = pickle.load(handle)
        if loaded.sklearn_version and loaded.sklearn_version != sklearn.__version__:
            warnings.warn(f"filter saved with scikit-learn {loaded.sklearn_version}, running {sklearn.__version__}: "
                          f"refit it from the validation maps (fit_filter) if scores look wrong")
        return loaded


def predict_scores(models: Sequence, features: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Mean prediction of the fold models, clipped to [0, 1]."""
    return np.clip(np.mean([m.predict(features[list(names)]) for m in models], axis=0), 0.0, 1.0)


# ----------------------------------------------------------------------------------------------------------------
# Fitting on validation maps
# ----------------------------------------------------------------------------------------------------------------

MapSource = Mapping[str, tuple[np.ndarray, np.ndarray]]   # filename -> (mask probability, spine probability)


def candidate_table(maps: MapSource, index: AnnotationIndex, decode_cfg: DecodeConfig = DecodeConfig(),
                    progress: Callable | None = None) -> pd.DataFrame:
    """One row per decoded validation instance: features + `match`, the share of the image's annotators with a
    filament at IoU > match_iou against it (the regression target)."""
    rows = []
    for filename in _iterate(maps, progress, "filter table"):
        probability, spine = maps[filename]
        instances = decode_instances(probability, decode_cfg)
        if not instances:
            continue
        gt_sets = ground_truth_sets(index, filename)
        matched = np.zeros(len(instances))
        for gt_masks in gt_sets:
            if gt_masks:
                matched += pairwise_iou(gt_masks, instances).max(axis=0) > decode_cfg.match_iou
        spine_max = float(spine.max())
        for k, mask in enumerate(instances):
            rows.append({"filename": filename, "instance": k, "match": float(matched[k] / max(len(gt_sets), 1)),
                         **instance_features(probability, mask, spine, spine_max)})
    return pd.DataFrame(rows, columns=["filename", "instance", "match", *FEATURES])


def fit_regressor(table: pd.DataFrame, cfg: FilterConfig = FilterConfig(), names: Sequence[str] = FEATURES):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                          max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
                                          random_state=cfg.seed)
    return model.fit(table[list(names)], table["match"])


def cross_fit(table: pd.DataFrame, cfg: FilterConfig = FilterConfig(),
              names: Sequence[str] = FEATURES) -> tuple[np.ndarray, list]:
    """Out-of-fold scores and the fold models, GroupKFold by observation date."""
    from sklearn.model_selection import GroupKFold

    groups = table["filename"].astype(str).map(observation_date)
    scores, models = np.zeros(len(table)), []
    for train, test in GroupKFold(n_splits=min(cfg.n_splits, groups.nunique())).split(table, groups=groups):
        model = fit_regressor(table.iloc[train], cfg, names)
        models.append(model)
        scores[test] = predict_scores([model], table.iloc[test], names)
    return scores, models


def evaluate_maps(maps: MapSource, index: AnnotationIndex, decode_cfg: DecodeConfig = DecodeConfig(),
                  keep: Callable[[str, list], list] | None = None, progress: Callable | None = None) -> pd.DataFrame:
    """Score rows (metrics.score_image) for every image; `keep(filename, instances)` optionally filters."""
    rows = []
    for filename in _iterate(maps, progress, "evaluate"):
        instances = decode_instances(maps[filename][0], decode_cfg)
        if keep is not None:
            instances = keep(filename, instances)
        rows.extend(score_image(filename, instances, index, decode_cfg.match_iou))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def threshold_curve(maps: MapSource, index: AnnotationIndex, table: pd.DataFrame, scores: np.ndarray,
                    thresholds: Sequence[float], decode_cfg: DecodeConfig = DecodeConfig()) -> pd.DataFrame:
    """Pooled PQ when instances scoring below each threshold are dropped (0 keeps everything). Best first."""
    by_image = {f: g.sort_values("instance")["score"].to_numpy()
                for f, g in table.assign(score=scores).groupby("filename")}
    rows: dict[float, list] = {t: [] for t in thresholds}
    for filename in maps:
        instances = decode_instances(maps[filename][0], decode_cfg)
        image_scores = by_image.get(filename, np.zeros(0))
        gt_sets = ground_truth_sets(index, filename)
        for t, bucket in rows.items():
            kept = [m for m, s in zip(instances, image_scores) if s >= t]
            bucket.extend(score_image(filename, kept, index, decode_cfg.match_iou, gt_sets))
    curve = pd.DataFrame([{"threshold": t, **aggregate_pq(pd.DataFrame(r, columns=ROW_COLUMNS))}
                          for t, r in rows.items()])
    return curve.sort_values("pq", ascending=False).reset_index(drop=True)


def closed_form_threshold(pq: float, sq: float) -> float:
    """Drop an instance iff its match probability p < 0.5 * PQ / SQ (see the module docstring)."""
    return float(0.5 * pq / sq) if sq > 0 else 0.0


def fit_filter(maps: MapSource, index: AnnotationIndex, decode_cfg: DecodeConfig = DecodeConfig(),
               cfg: FilterConfig = FilterConfig(), log: Callable = print,
               progress: Callable | None = None) -> InstanceFilter | None:
    """The full recipe in the module docstring. Returns None when no threshold beats keeping everything."""
    table = candidate_table(maps, index, decode_cfg, progress)
    scores, models = cross_fit(table, cfg)
    unfiltered = aggregate_pq(evaluate_maps(maps, index, decode_cfg, progress=progress))
    anchor = closed_form_threshold(unfiltered["pq"], unfiltered["sq"])
    grid = sorted({0.0, *[round(max(0.0, anchor + d), 4) for d in cfg.grid_offsets]})
    curve = threshold_curve(maps, index, table, scores, grid, decode_cfg)
    best = float(curve["threshold"].iloc[0])
    near = curve.loc[(curve["threshold"] - anchor).abs().idxmin()]
    if abs(float(curve["pq"].iloc[0]) - float(near["pq"])) < cfg.near_tie:
        best = float(near["threshold"])
    pq_best = float(curve.loc[curve["threshold"] == best, "pq"].iloc[0])
    pq_none = float(curve.loc[curve["threshold"] == 0.0, "pq"].iloc[0])
    report = {"instances": len(table), "unfiltered_pq": pq_none, "unfiltered_sq": float(unfiltered["sq"]),
              "closed_form": anchor, "threshold": best, "filtered_pq": pq_best,
              "auc_matched_any": _auc(table["match"] > 0, scores), "curve": curve.round(5).to_dict("records")}
    if best <= 0 or pq_best <= pq_none:
        log(f"filter does not beat unfiltered PQ {pq_none:.5f}: left off")
        return None
    drop_rate = float((scores < best).mean())
    deployed = predict_scores(models, table, FEATURES)
    threshold = float(np.quantile(deployed, drop_rate)) if drop_rate > 0 else 0.0
    report.update(drop_rate=drop_rate, deploy_threshold=threshold)
    log(f"filter: unfiltered PQ {pq_none:.5f} -> filtered {pq_best:.5f} at t={best:.4f} (closed form {anchor:.4f}); "
        f"drops {drop_rate:.1%} -> deployed threshold {threshold:.4f}")
    oof = table[["filename", "instance", "match"]].assign(score=scores)
    return InstanceFilter(models, threshold, list(FEATURES), report, oof=oof)


def _auc(labels: pd.Series, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else float("nan")


def _iterate(maps: MapSource, progress: Callable | None, desc: str):
    return progress(list(maps), desc=desc) if progress is not None else list(maps)
