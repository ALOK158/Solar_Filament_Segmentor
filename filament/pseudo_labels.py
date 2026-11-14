"""Teacher-student training on unlabelled GONG frames (the best use of unlabelled data we measured).

1. TEACHER: the deployed 3-member ensemble predicts soft mask + spine maps for every unlabelled frame (4-flip TTA),
   stored as uint8 at the student's input size. The maps stay soft: the teacher's estimate of the chance that an
   annotator draws a pixel, which carries its uncertainty into the student.
2. PRIOR SHIFT: the unlabelled frames are 2023-2026, near solar maximum: the teacher finds a median of 14
   filaments per frame against 9 on the 2011-2022 validation images. Each frame is weighted by
   (validation share of its filament-count bin) / (unlabelled share of that bin), clipped to [0.2, 5], and frames
   are drawn with those weights.
3. STUDENT: warm-started from v24, each epoch sees the 566 labelled images plus 566 weighted pseudo-labelled frames
   (config.RECIPES["student"]).

Result (fold 0): student 0.4487 vs 0.4347 for the identical schedule without pseudo frames
(+0.014, P 1.00). The gain is in calibration: the filter's AUC rose 0.792 -> 0.810. One student equals the
3-model teacher (-0.0009, P 0.42) at one third of the inference cost; adding it to the teacher's ensemble does not
help (-0.009), because a student distils the same direction. Public LB 0.39 for both.

Data rules: unlabelled frames are used for TRAINING only and come from dates after the labelled set (none within
an hour of a test image); inference never sees anything but the test image itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .config import DecodeConfig, InferenceConfig, InputConfig
from .data import read_gray
from .decode import decode_instances
from .inference import Model, predict_small, quantise, to_native
from .preprocess import network_input

COUNT_BINS = (0, 2, 4, 6, 8, 10, 13, 16, 20, 10 ** 6)


def teacher_label(models: Sequence[Model], frames: Sequence[str | Path], out_dir: str | Path,
                  device: torch.device | str, input_cfg: InputConfig = InputConfig(),
                  inference_cfg: InferenceConfig = InferenceConfig(), decode_cfg: DecodeConfig = DecodeConfig(),
                  progress: Callable | None = None) -> pd.DataFrame:
    """Soft teacher maps for each frame -> <out_dir>/<stem>.npz ('p' mask, 's' spine, uint8 at the input size).
    Returns one row per frame: frame path, maps path, the teacher's instance count (native-resolution decode)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in (progress(list(frames), desc="teacher") if progress else frames):
        image = read_gray(frame)
        small = predict_small(models, network_input(image, input_cfg), device, inference_cfg)
        count = len(decode_instances(to_native(small[:1], *image.shape)[0], decode_cfg))
        target = out_dir / f"{Path(frame).stem}.npz"
        np.savez_compressed(target, p=quantise(small[0]), s=quantise(small[1]))
        rows.append({"frame": str(frame), "maps": str(target), "teacher_instances": count})
    return pd.DataFrame(rows)


def prior_shift_weights(frame_counts: Sequence[int], reference_counts: Sequence[int],
                        bins: Sequence[float] = COUNT_BINS, clip: tuple[float, float] = (0.2, 5.0)) -> np.ndarray:
    """Importance weight per frame: reference share / frame share of its instance-count bin (+0.5 smoothing)."""
    bins = np.asarray(bins)
    h_ref = np.histogram(reference_counts, bins)[0] + 0.5
    h_frames = np.histogram(frame_counts, bins)[0] + 0.5
    ratio = (h_ref / h_ref.sum()) / (h_frames / h_frames.sum())
    which = np.clip(np.digitize(frame_counts, bins) - 1, 0, len(bins) - 2)
    return np.clip(ratio[which], *clip)


def effective_sample_size(weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def reference_counts(maps: Mapping[str, tuple[np.ndarray, np.ndarray]], decode_cfg: DecodeConfig = DecodeConfig()) -> list[int]:
    """The teacher's instance counts on the validation maps: the era the weights shift the frames toward."""
    return [len(decode_instances(maps[f][0], decode_cfg)) for f in maps]
