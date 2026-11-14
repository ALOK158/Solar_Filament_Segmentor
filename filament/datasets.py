"""PyTorch datasets for training: labelled whole-disk images, pseudo-labelled unlabelled frames, and the sampler
that mixes them per epoch.

Every sample is the WHOLE disk resized to the input size (1536 px), never a crop. Augmentation randomness is
seeded by (recipe seed, epoch, sample index), so a training run draws the same samples and augmentations on
every machine (GPU arithmetic still differs in the last bits).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment
from .config import InputConfig, TrainConfig
from .data import AnnotationIndex, read_gray, record_spine_mask, record_union_mask
from .preprocess import resize_image, resize_mask, to_channels


def _item(image: np.ndarray, mask: np.ndarray, spine: np.ndarray, clahe_clip: float, input_cfg: InputConfig,
          pseudo: bool) -> dict[str, torch.Tensor]:
    """Tensors for one sample. Components are labelled AFTER augmentation so their ids match the final mask."""
    _, components = cv2.connectedComponents(np.ascontiguousarray(mask >= 0.5).astype(np.uint8), connectivity=8)
    return {"image": torch.from_numpy(to_channels(image, input_cfg, clahe_clip=clahe_clip)),
            "target": torch.from_numpy(np.ascontiguousarray(mask).astype(np.float32)[None]),
            "spine": torch.from_numpy(np.ascontiguousarray(spine).astype(np.float32)[None]),
            "components": torch.from_numpy(components.astype(np.int32)[None]),
            "pseudo": torch.tensor(int(pseudo))}


class SeededDataset(Dataset):
    """Base class: `epoch` is set by the trainer before each epoch (workers are re-created every epoch)."""
    epoch = 0

    def rng(self, seed: int, index: int, train: bool) -> np.random.Generator:
        return np.random.default_rng((seed, self.epoch if train else 0, index))


class LabelledDataset(SeededDataset):
    """Annotated images. The target record is the medoid annotator, or (with probability
    cfg.random_annotator_prob) a random annotator of that image: the other annotators' drawings used as label
    augmentation."""

    def __init__(self, filenames: Sequence[str], image_dir: str | Path, index: AnnotationIndex,
                 medoid: Mapping[str, str], cfg: TrainConfig, train: bool, input_cfg: InputConfig | None = None):
        self.filenames, self.image_dir, self.index, self.medoid = list(filenames), Path(image_dir), index, medoid
        self.cfg, self.train = cfg, train
        self.input_cfg = input_cfg or InputConfig(input_size=cfg.input_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def record_for(self, filename: str, rng: np.random.Generator) -> str:
        p = self.cfg.random_annotator_prob
        if self.train and p > 0 and (p >= 1.0 or rng.random() < p):
            records = self.index.records_by_file[filename]
            return records[int(rng.integers(len(records)))]
        return self.medoid[filename]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        filename, size = self.filenames[i], self.input_cfg.input_size
        rng = self.rng(self.cfg.seed, i, self.train)
        record = self.record_for(filename, rng)
        target = np.stack([record_union_mask(self.index, record).astype(np.uint8),
                           record_spine_mask(self.index, record, self.cfg.spine_tube_px).astype(np.uint8)], axis=-1)
        image = resize_image(read_gray(self.image_dir / filename), size)
        target = resize_mask(target, size)
        clahe_clip = self.input_cfg.clahe_clip
        if self.train:
            image, clahe_clip, target = augment(image, target, rng, self.cfg)
        return _item(image, target[..., 0], target[..., 1], clahe_clip, self.input_cfg, pseudo=False)


class PseudoLabelDataset(SeededDataset):
    """Unlabelled frames with SOFT teacher targets: `items` = (frame path, teacher .npz with uint8 'p' mask and
    's' spine maps at the input size). The targets stay probabilities: the teacher's estimate of the chance that an
    annotator draws each pixel."""

    def __init__(self, items: Sequence[tuple[str | Path, str | Path]], cfg: TrainConfig,
                 input_cfg: InputConfig | None = None):
        self.items = [(Path(a), Path(b)) for a, b in items]
        self.cfg = cfg
        self.input_cfg = input_cfg or InputConfig(input_size=cfg.input_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        frame, maps = self.items[i]
        image = resize_image(read_gray(frame), self.input_cfg.input_size)
        with np.load(maps) as z:
            target = np.stack([z["p"], z["s"]], axis=-1).astype(np.float32) / 255.0
        if target.shape[:2] != image.shape:
            raise ValueError(f"{maps}: teacher maps {target.shape[:2]} do not match the input {image.shape}")
        image, clahe_clip, target = augment(image, target, self.rng(self.cfg.seed + 1, i, True), self.cfg)
        return _item(image, target[..., 0], target[..., 1], clahe_clip, self.input_cfg, pseudo=True)


class MixedEpochSampler(torch.utils.data.Sampler):
    """Each epoch: every labelled index `labelled_repeat` times plus `extra_per_epoch` pseudo-labelled indices drawn
    without replacement with probabilities proportional to `extra_weights`, all shuffled together. Indices
    >= n_labelled address the pseudo dataset inside a ConcatDataset."""

    def __init__(self, n_labelled: int, labelled_repeat: int = 1, n_extra: int = 0, extra_per_epoch: int = 0,
                 extra_weights: Sequence[float] | None = None, seed: int = 0):
        self.n_labelled, self.repeat, self.n_extra = n_labelled, max(labelled_repeat, 0), n_extra
        self.per_epoch = min(extra_per_epoch, n_extra) if n_extra else 0
        weights = np.ones(n_extra) if extra_weights is None else np.asarray(extra_weights, dtype=float)
        self.p = weights / weights.sum() if n_extra else None
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.n_labelled * self.repeat + self.per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 7919 * self.epoch)
        self.epoch += 1
        order = np.tile(np.arange(self.n_labelled), self.repeat)
        if self.per_epoch:
            pick = rng.choice(self.n_extra, size=self.per_epoch, replace=False, p=self.p) + self.n_labelled
            order = np.concatenate([order, pick])
        rng.shuffle(order)
        return iter(int(i) for i in order)
