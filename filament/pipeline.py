"""End-to-end helpers the notebook calls: the deployed segmenter, validation-map prediction, the validation
report, and the environment report."""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from .config import SystemConfig
from .data import AnnotationIndex, ground_truth_sets, list_images, read_gray
from .decode import decode_instances
from .inference import Model, predict_maps, predict_small, quantise, to_native
from .instance_filter import InstanceFilter
from .maps import MapStore, write_map_store
from .metrics import AgreementStats, ROW_COLUMNS, aggregate_pq, human_agreement_rows, pairwise_iou, score_image
from .preprocess import network_input, to_channels
from .refiner import refine
from .submission import image_rows, write_submission
from .timing import StageTimer


@dataclass
class ImageResult:
    filename: str
    instances: list            # kept instance masks, submission order
    scores: np.ndarray         # their filter scores
    candidates: list           # every decoded candidate
    candidate_scores: np.ndarray
    maps: np.ndarray | None = None   # uint8 (2, H, W) probability maps, when kept for figures


class Segmenter:
    """The deployed system: members -> TTA probability maps -> decode -> filter -> refine (when refiners are given)."""

    def __init__(self, models: Sequence[Model], instance_filter: InstanceFilter | None, device: torch.device | str,
                 cfg: SystemConfig = SystemConfig(), timer: StageTimer | None = None,
                 refiners: Sequence[torch.nn.Module] = ()) -> None:
        self.models, self.filter, self.device, self.cfg = list(models), instance_filter, device, cfg
        self.refiners = list(refiners)
        self.timer = timer or StageTimer(device)

    def maps(self, image: np.ndarray) -> np.ndarray:
        """float32 (2, H, W): filament and spine probability at native resolution."""
        with self.timer("preprocess"):
            channels = network_input(image, self.cfg.input)
        with self.timer("network", sync=True):
            small = predict_small(self.models, channels, self.device, self.cfg.inference)
        with self.timer("upsample"):
            return to_native(small, *image.shape)

    def segment(self, filename: str, image: np.ndarray, keep_maps: bool = False) -> ImageResult:
        maps = self.maps(image)
        with self.timer("decode"):
            candidates = decode_instances(maps[0], self.cfg.decode)
        with self.timer("filter"):
            if self.filter is not None:
                kept, scores, all_scores = self.filter.apply(maps[0], maps[1], candidates)
            else:
                kept, scores, all_scores = candidates, np.ones(len(candidates)), np.ones(len(candidates))
        if self.refiners and kept:
            with self.timer("refine", sync=True):
                kept, index = refine(self.refiners, to_channels(image, self.cfg.input), maps[0], maps[1], kept,
                                     self.device, self.cfg.refine)
                scores = np.asarray(scores)[index]
        return ImageResult(filename, kept, scores, candidates, all_scores, quantise(maps) if keep_maps else None)

    def segment_folder(self, folder: str | Path, out_csv: str | Path, keep: Sequence[str] = (),
                       progress: Callable | None = None, limit: int | None = None):
        """Every image of `folder` -> submission CSV. Returns (submission, one row per image, {name: ImageResult}
        for the names in `keep`)."""
        rows, records, examples = [], [], {}
        paths = list_images(folder)[:limit] if limit else list_images(folder)
        for path in (progress(paths, desc="test images") if progress else paths):
            with self.timer("read"):
                image = read_gray(path)
            result = self.segment(path.name, image, keep_maps=path.name in keep)
            with self.timer("encode"):
                rows.extend(image_rows(path.stem, result.instances))
            records.append({"filename": path.name, "candidates": len(result.candidates), "kept": len(result.instances)})
            if path.name in keep:
                examples[path.name] = result
        return write_submission(rows, out_csv), pd.DataFrame(records), examples


def predict_validation_maps(models: Sequence[Model], filenames: Sequence[str], image_dir: str | Path,
                            out_path: str | Path, device: torch.device | str, cfg: SystemConfig = SystemConfig(),
                            progress: Callable | None = None) -> MapStore:
    """Ensemble maps of the validation images, stored as uint8 in one compressed .npz (see maps.MapStore)."""
    def generate():
        for filename in (progress(list(filenames), desc="validation maps") if progress else filenames):
            image = read_gray(Path(image_dir) / filename)
            yield filename, quantise(predict_maps(models, image, device, cfg.input, cfg.inference))
    return MapStore(write_map_store(out_path, generate()))


@dataclass
class ValidationReport:
    unfiltered: dict
    filtered: dict
    human: dict
    ours_stats: dict
    human_stats: dict
    per_image: pd.DataFrame
    pairs: dict = field(default_factory=dict)      # IoU / Dice of all overlapping pairs, ours and human
    human_rows: pd.DataFrame | None = None         # human-vs-human score rows (for bootstrap intervals)


def refine_validation(maps: MapStore, image_dir: str | Path, instance_filter: InstanceFilter,
                      refiners: Sequence[torch.nn.Module], device: torch.device | str,
                      cfg: SystemConfig = SystemConfig(), progress: Callable | None = None) -> dict[str, list]:
    """The fold-0 instances the filter keeps (out-of-fold decisions), refined; filename -> list of COCO RLEs
    (948 full-size masks would not fit in memory)."""
    refined = {}
    for filename in (progress(list(maps), desc="refine validation") if progress else maps):
        kept, _ = kept_validation_instances(maps, filename, instance_filter, cfg)
        masks = []
        if kept:
            probability, spine = maps[filename]
            channels = to_channels(read_gray(Path(image_dir) / filename), cfg.input)
            masks, _ = refine(refiners, channels, probability, spine, kept, device, cfg.refine)
        refined[filename] = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    return refined


def validate_system(maps: MapStore, index: AnnotationIndex, instance_filter: InstanceFilter,
                    cfg: SystemConfig = SystemConfig(), progress: Callable | None = None,
                    refined: Mapping[str, list] | None = None) -> ValidationReport:
    """Fold-0 numbers of the deployed system. Each candidate is kept or dropped by its OUT-OF-FOLD filter score at
    the cross-validated threshold, so no number below has seen its own image during filter fitting. `refined`
    (from refine_validation) replaces the kept instances by their refined masks; 'filtered' then means the full
    system and 'unfiltered' is unchanged."""
    oof = {f: g.sort_values("instance")["score"].to_numpy() for f, g in instance_filter.oof.groupby("filename")}
    threshold = float(instance_filter.report["threshold"])
    rows_raw, rows_kept, per_image = [], [], []
    ours, human = AgreementStats(), AgreementStats()
    for filename in (progress(list(maps), desc="validation") if progress else maps):
        candidates = decode_instances(maps[filename][0], cfg.decode)
        scores = oof.get(filename, np.zeros(0))
        kept = [m for m, s in zip(candidates, scores) if s >= threshold]
        if refined is not None:
            kept = [mask_utils.decode(r).astype(bool) for r in refined.get(filename, [])]
        gt_sets = ground_truth_sets(index, filename)
        image_rows = score_image(filename, kept, index, cfg.decode.match_iou, gt_sets)
        rows_raw += score_image(filename, candidates, index, cfg.decode.match_iou, gt_sets)
        rows_kept += image_rows
        marks = (len(ours.gt_degree), len(ours.pred_degree), len(human.gt_degree), len(human.pred_degree))
        for gt in gt_sets:
            ours.add(gt, kept)
        for p, predicted in enumerate(gt_sets):
            for a, truth in enumerate(gt_sets):
                if a != p:
                    human.add(truth, predicted)
        agg = aggregate_pq(image_rows)
        per_image.append({"filename": filename, "annotators": len(gt_sets), "candidates": len(candidates),
                          "kept": len(kept), "pq": agg["pq"], "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
                          "sum_iou": float(sum(r["sum_iou"] for r in image_rows)),
                          **_degree_counts("ours", ours.gt_degree[marks[0]:], ours.pred_degree[marks[1]:]),
                          **_degree_counts("human", human.gt_degree[marks[2]:], human.pred_degree[marks[3]:])})
    human_rows = human_agreement_rows(index, list(maps), cfg.decode.match_iou)
    return ValidationReport(aggregate_pq(pd.DataFrame(rows_raw, columns=ROW_COLUMNS)),
                            aggregate_pq(pd.DataFrame(rows_kept, columns=ROW_COLUMNS)), aggregate_pq(human_rows),
                            ours.summary(), human.summary(), pd.DataFrame(per_image),
                            {"ours_iou": ours.iou, "ours_dice": ours.dice, "human_iou": human.iou,
                             "human_dice": human.dice}, human_rows)


def _degree_counts(prefix: str, gt_degree: Sequence[int], pred_degree: Sequence[int]) -> dict:
    """Per-image counts behind the rubric rates, so they can be bootstrapped over images."""
    gt, pred = np.asarray(gt_degree, dtype=int), np.asarray(pred_degree, dtype=int)
    return {f"{prefix}_gt": len(gt), f"{prefix}_missed": int((gt == 0).sum()), f"{prefix}_fragmented": int((gt >= 2).sum()),
            f"{prefix}_pred": len(pred), f"{prefix}_false_alarm": int((pred == 0).sum()),
            f"{prefix}_merged": int((pred >= 2).sum())}


def kept_validation_instances(maps: MapStore, filename: str, instance_filter: InstanceFilter,
                              cfg: SystemConfig = SystemConfig()) -> tuple[list, list]:
    """(kept, dropped) candidates of one validation image under the out-of-fold scores."""
    candidates = decode_instances(maps[filename][0], cfg.decode)
    scores = instance_filter.oof.loc[instance_filter.oof["filename"] == filename].sort_values("instance")["score"]
    keep = scores.to_numpy() >= float(instance_filter.report["threshold"])
    return [m for m, k in zip(candidates, keep) if k], [m for m, k in zip(candidates, keep) if not k]


def match_matrices(gt_sets: Sequence[Sequence[np.ndarray]], predicted: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [pairwise_iou(list(gt), list(predicted)) for gt in gt_sets]


def environment() -> dict:
    """Versions and hardware, printed at the top of the notebook and stored with the results."""
    import cv2, sklearn, timm, scipy, pycocotools  # noqa: E401

    report = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
              "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
              "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
              "pycocotools": getattr(pycocotools, "__version__", "installed"),
              "cuda": torch.version.cuda if torch.cuda.is_available() else None,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cpu_threads": torch.get_num_threads()}
    try:
        import onnx, onnxruntime  # noqa: E401
        report.update(onnx=onnx.__version__, onnxruntime=onnxruntime.__version__,
                      onnxruntime_providers=onnxruntime.get_available_providers())
    except ImportError:
        report.update(onnx=None, onnxruntime=None)
    return report
