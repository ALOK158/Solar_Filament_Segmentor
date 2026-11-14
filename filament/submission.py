"""Submission format and the host's checks.

One row per filament: filament_id = <image stem>_<k> (k from 1), segmentation_rle = the pycocotools compressed
RLE counts string of the 2048 x 2048 mask. The host rejects overlapping instances and keys images by the text
before the FIRST underscore of filament_id, so stems must not contain underscores (GONG stems never do).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .data import list_images

COLUMNS = ["filament_id", "segmentation_rle"]


def encode_rle(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def decode_rle(counts: str, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    return mask_utils.decode({"size": list(shape), "counts": counts.encode("ascii")}).astype(bool)


def image_rows(stem: str, instances: Sequence[np.ndarray]) -> list[dict]:
    return [{"filament_id": f"{stem}_{k}", "segmentation_rle": encode_rle(m)} for k, m in enumerate(instances, 1)]


def write_submission(rows: list[dict], path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame["filament_id"].is_unique:
        raise ValueError("duplicate filament_id")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def check_submission(frame: pd.DataFrame, test_dir: str | Path, shape: tuple[int, int] = (2048, 2048)) -> dict:
    """Everything the host checks, asserted before spending a submission: unique ids, every id maps to a test
    image under the host's own parsing, RLEs decode to the image size, and no pixel belongs to two instances."""
    stems = {p.stem for p in list_images(test_dir)}
    assert stems, f"no test images under {test_dir}"
    assert frame["filament_id"].is_unique, "duplicate filament_id"
    keys = frame["filament_id"].str.split("_", n=1).str[0]
    orphans = sorted(set(keys) - stems)
    assert not orphans, f"{len(orphans)} ids match no test image, e.g. {orphans[:3]}"
    worst = 0
    for stem, group in frame.groupby(keys):
        count = np.zeros(shape, dtype=np.uint8)
        for counts in group["segmentation_rle"]:
            count += decode_rle(counts, shape)
        worst = max(worst, int(count.max()))
    assert worst <= 1, "overlapping instances: the host rejects this file"
    return {"rows": len(frame), "test_images": len(stems), "images_with_filaments": int(keys.nunique()),
            "images_without": len(stems) - int(keys.nunique()), "max_pixel_multiplicity": worst}
