"""Competition files, COCO annotations, annotator records and validation folds.

MAGFiLO stores one COCO "image" entry per (image, annotator) pair, which we call a RECORD: 707 training images carry
1154 records (411 images with one annotator, 145 with two, 151 with three). The metric scores a prediction against
every record of an image and pools the counts, so most functions here work per record.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

TRAIN_JSON_GLOB = "MAGFiLO*_train.json"
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


@dataclass(frozen=True)
class CompetitionPaths:
    root: Path
    annotations: Path
    train_images: Path
    test_images: Path


def find_competition_data(root: str | Path | None = None) -> CompetitionPaths:
    """Locate the competition files. Looks in `root`, then $FILAMENT_DATA, /kaggle/input, ./data and ../data.

    Works for the Kaggle layout (<root>/train/train_images, <root>/test/test_images) and for a flat local copy
    (<root>/train_images, <root>/test_images): the annotation JSON is found first, then the two image folders by
    name anywhere below the directory that holds it (or its parent)."""
    candidates = [root, os.environ.get("FILAMENT_DATA"), "/kaggle/input", "data", "../data"]
    for candidate in [Path(c) for c in candidates if c]:
        if not candidate.is_dir():
            continue
        found = sorted(candidate.rglob(TRAIN_JSON_GLOB))
        if not found:
            continue
        annotations = found[0]
        for base in (annotations.parent, annotations.parent.parent):
            train = _find_dir(base, "train_images")
            test = _find_dir(base, "test_images")
            if train and test:
                return CompetitionPaths(base, annotations, train, test)
    raise FileNotFoundError("competition data not found: pass the folder holding MAGFiLO*_train.json, "
                            "train_images/ and test_images/, or set FILAMENT_DATA")


def _find_dir(base: Path, name: str) -> Path | None:
    direct = base / name
    if direct.is_dir():
        return direct
    hits = sorted(p for p in base.rglob(name) if p.is_dir())
    return hits[0] if hits else None


def list_images(folder: str | Path) -> list[Path]:
    """Every image in `folder`, whatever the spelling of its extension, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_gray(path: str | Path) -> np.ndarray:
    """uint8 (H, W). GONG H-alpha frames are single-channel JPEGs."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def observation_date(filename: str) -> str:
    """YYYYMMDD. File names are <YYYYMMDDhhmmss><site>h.jpeg, e.g. 20140609195854Bh.jpeg."""
    return Path(filename).name[:8]


def site_code(filename: str) -> str:
    """The GONG station letter (B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur)."""
    return Path(filename).stem[14:15]


# ----------------------------------------------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class AnnotationIndex:
    """COCO annotations indexed three ways. Record ids are COCO image ids (one per image and annotator)."""
    records: dict[str, dict]                 # record id -> COCO image entry (file_name, height, width, ...)
    annotations: dict[str, list[dict]]       # record id -> its filament annotations
    records_by_file: dict[str, list[str]]    # file name -> its record ids, sorted

    @property
    def filenames(self) -> list[str]:
        return sorted(self.records_by_file)


def load_annotations(path: str | Path) -> AnnotationIndex:
    with Path(path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    records = {str(entry["id"]): entry for entry in coco["images"]}
    annotations: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(str(annotation["image_id"]), []).append(annotation)
    records_by_file: dict[str, list[str]] = {}
    for record_id, entry in records.items():
        records_by_file.setdefault(entry["file_name"], []).append(record_id)
    for ids in records_by_file.values():
        ids.sort()
    return AnnotationIndex(records, annotations, records_by_file)


def _filaments(index: AnnotationIndex, record_id: str) -> list[dict]:
    return [a for a in index.annotations.get(record_id, []) if not a.get("iscrowd", 0)]


def _shape(index: AnnotationIndex, record_id: str) -> tuple[int, int]:
    entry = index.records[record_id]
    return int(entry["height"]), int(entry["width"])


def annotation_rle(annotation: dict, height: int, width: int) -> dict:
    """COCO polygon(s) or RLE -> one compressed RLE."""
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        return mask_utils.frPyObjects(rle, height, width) if isinstance(rle.get("counts"), list) else rle
    if not segmentation:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    return mask_utils.decode(annotation_rle(annotation, height, width)).astype(bool)


def record_masks(index: AnnotationIndex, record_id: str) -> list[np.ndarray]:
    """One boolean mask per filament drawn in this record (native resolution)."""
    height, width = _shape(index, record_id)
    return [annotation_mask(a, height, width) for a in _filaments(index, record_id)]


def record_union_rle(index: AnnotationIndex, record_id: str) -> dict:
    height, width = _shape(index, record_id)
    rles = [annotation_rle(a, height, width) for a in _filaments(index, record_id)]
    if not rles:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    return mask_utils.merge(rles)


def record_union_mask(index: AnnotationIndex, record_id: str) -> np.ndarray:
    """The semantic training target: the union of a record's filaments."""
    return mask_utils.decode(record_union_rle(index, record_id)).astype(bool)


def record_spine_mask(index: AnnotationIndex, record_id: str, tube_px: int = 3) -> np.ndarray:
    """The record's human-drawn spine polylines rendered as a `tube_px`-wide tube.

    MAGFiLO ships a drawn centre line per filament. It is the structure's real axis, unlike a morphological
    skeleton of the mask, which sprouts a branch down every barb."""
    height, width = _shape(index, record_id)
    canvas = np.zeros((height, width), dtype=np.uint8)
    for annotation in _filaments(index, record_id):
        points = np.asarray(annotation.get("spine") or [], dtype=np.float32).reshape(-1, 2)
        if len(points) >= 2:
            cv2.polylines(canvas, [np.round(points).astype(np.int32)], False, 1, thickness=tube_px)
    return canvas.astype(bool)


def medoid_records(index: AnnotationIndex) -> dict[str, str]:
    """Per image, the record whose union mask has the highest mean IoU with the other records of that image.

    The metric pools all annotators, so the most typical annotator is the best single training target (it also
    avoids the bias of picking the largest or smallest drawing)."""
    selected = {}
    for filename, record_ids in index.records_by_file.items():
        if len(record_ids) == 1:
            selected[filename] = record_ids[0]
            continue
        unions = [record_union_rle(index, r) for r in record_ids]
        iou = np.asarray(mask_utils.iou(unions, unions, [0] * len(unions)))
        agreement = (iou.sum(axis=1) - 1.0) / (len(record_ids) - 1)
        selected[filename] = record_ids[int(np.argmax(agreement))]
    return selected


def ground_truth_sets(index: AnnotationIndex, filename: str) -> list[list[np.ndarray]]:
    """Every record of an image as a list of filament masks: what a prediction is scored against."""
    return [record_masks(index, record_id) for record_id in index.records_by_file[filename]]


# ----------------------------------------------------------------------------------------------------------------
# Folds and summaries
# ----------------------------------------------------------------------------------------------------------------

def load_folds(path: str | Path) -> pd.DataFrame:
    """filename, site, n_annotators, mean_instances, max_instances, stratum, fold.

    Five folds stratified by site x filament-count quartile. Fold 0 (141 images) is the validation fold of every
    number reported for the system; folds 1-4 (566 images) are the training set."""
    return pd.read_csv(path)


def image_table(index: AnnotationIndex, filenames: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per image: site, year, annotator count, filaments per annotator (mean / min / max)."""
    rows = []
    for filename in (filenames if filenames is not None else index.filenames):
        counts = [len(_filaments(index, r)) for r in index.records_by_file[filename]]
        rows.append({"filename": filename, "site": site_code(filename), "year": int(filename[:4]),
                     "annotators": len(counts), "filaments_mean": float(np.mean(counts)),
                     "filaments_min": int(min(counts)), "filaments_max": int(max(counts))})
    return pd.DataFrame(rows)


def annotation_metadata(index: AnnotationIndex) -> dict[str, Any]:
    """Counts for the data-audit table."""
    table = image_table(index)
    return {"images": len(table), "records": len(index.records),
            "filaments": int(sum(len(_filaments(index, r)) for r in index.records)),
            "images_by_annotators": table["annotators"].value_counts().sort_index().to_dict(),
            "sites": table["site"].value_counts().to_dict(),
            "years": f"{table['year'].min()}-{table['year'].max()}"}
