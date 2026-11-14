"""Unit tests for scoring, decoding, refiner geometry and paste-back, and submission checks (no data or GPU needed):
pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import decode, instance_filter, metrics, submission
from filament.config import DecodeConfig


def disk_mask(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def test_rle_roundtrip():
    mask = disk_mask((64, 64), (20, 30), 9)
    assert np.array_equal(submission.decode_rle(submission.encode_rle(mask), mask.shape), mask)


def test_pq_perfect_and_empty():
    a = [disk_mask((64, 64), (20, 20), 8)]
    assert metrics.panoptic_quality(a, a)["pq"] == pytest.approx(1.0)
    assert metrics.panoptic_quality([], [])["pq"] == 1.0
    assert metrics.panoptic_quality(a, [])["fn"] == 1


def test_pq_iou_gate_is_strict():
    gt = np.zeros((10, 10), bool); gt[:, :4] = True            # 40 px
    pred = np.zeros((10, 10), bool); pred[:, 2:6] = True       # IoU = 20 / 60 = 1/3 -> no match
    r = metrics.panoptic_quality([gt], [pred])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_pooled_pq_sums_counts_not_ratios():
    rows = pd.DataFrame([{"filename": "a", "annotator": 0, "pq": 1, "sq": 1, "rq": 1, "tp": 1, "fp": 0, "fn": 0,
                          "sum_iou": 1.0, "n_gt": 1, "n_pred": 1},
                         {"filename": "b", "annotator": 0, "pq": 0, "sq": 0, "rq": 0, "tp": 0, "fp": 3, "fn": 3,
                          "sum_iou": 0.0, "n_gt": 3, "n_pred": 3}])
    assert metrics.aggregate_pq(rows)["pq"] == pytest.approx(1.0 / (1 + 1.5 + 1.5))


def test_decode_threshold_area_and_disjoint():
    p = np.zeros((128, 128), np.float32)
    p[disk_mask(p.shape, (30, 30), 12)] = 0.9        # ~450 px: kept
    p[disk_mask(p.shape, (90, 90), 5)] = 0.9         # ~80 px: below min_area 240
    p[disk_mask(p.shape, (30, 90), 12)] = 0.55       # below threshold 0.60
    inst = decode.decode_instances(p, DecodeConfig())
    assert len(inst) == 1 and inst[0][30, 30]
    labels = decode.label_image(inst)
    assert labels.max() == 1


def test_closed_form_threshold():
    assert instance_filter.closed_form_threshold(0.45, 0.72) == pytest.approx(0.3125)


def test_check_submission_rejects_overlap(tmp_path):
    (tmp_path / "20200101000000Bh.jpeg").write_bytes(b"")
    m = disk_mask((2048, 2048), (500, 500), 20)
    ok = pd.DataFrame(submission.image_rows("20200101000000Bh", [m]))
    assert submission.check_submission(ok, tmp_path)["max_pixel_multiplicity"] == 1
    bad = pd.DataFrame(submission.image_rows("20200101000000Bh", [m, m]))
    with pytest.raises(AssertionError):
        submission.check_submission(bad, tmp_path)


def test_agreement_stats_counts_merges():
    a, b = np.zeros((40, 40), bool), np.zeros((40, 40), bool)
    a[5:10, 5:30] = True; b[12:17, 5:30] = True
    merged = a | b; merged[10:12, 5:30] = True
    stats = metrics.AgreementStats(); stats.add([a, b], [merged])
    s = stats.summary()
    assert s["merged n:1"] == 1.0 and s["missed 1:0"] == 0.0


def ribbon(shape, row, col0, col1, width):
    mask = np.zeros(shape, bool)
    mask[row:row + width, col0:col1] = True
    return mask


def test_refiner_crop_roundtrip():
    from filament import refiner
    mask = ribbon((300, 300), 100, 60, 220, 6)
    affine = refiner.crop_affine(mask)
    crop = refiner.warp([mask.astype(np.float32)], affine)[0]
    (y0, y1, x0, x1), region = refiner.unwarp(crop, affine, mask.shape)
    back = np.zeros(mask.shape, bool)
    back[y0:y1, x0:x1] = region >= 0.5
    assert (back & mask).sum() / (back | mask).sum() > 0.99      # native scale: the crop loses nothing


def test_refiner_paste_back_disjoint_and_seed_fallback():
    from filament import refiner
    from filament.config import RefinerConfig
    cfg = RefinerConfig(min_area=50)
    a, b = ribbon((300, 300), 100, 40, 200, 6), ribbon((300, 300), 104, 150, 280, 6)
    b &= ~a                                                       # seeds are disjoint, as decoded instances are
    affines = [refiner.crop_affine(m, cfg) for m in (a, b)]
    grown = [refiner.warp([np.maximum(m, ribbon(m.shape, 98, 30, 290, 12)).astype(np.float32)], t, cfg)[0]
             for m, t in zip((a, b), affines)]                   # both refinements claim the shared region
    out, index = refiner.paste_back([a, b], np.stack(grown), affines, cfg, return_index=True)
    assert sum(int(m.sum()) for m in out) == int(np.logical_or.reduce(out).sum())   # no pixel counted twice
    empty = np.zeros((1, cfg.height, cfg.width), np.float32)     # a failed refinement keeps the seed
    kept, idx = refiner.paste_back([a], empty, affines[:1], cfg, return_index=True)
    assert idx == [0] and np.array_equal(kept[0], a)
