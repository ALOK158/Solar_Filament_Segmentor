"""Every setting of the final system, as frozen dataclasses.

Defaults are the values the shipped models were trained, tuned and validated with. Each comment gives the reason or
the measurement behind the value; "fold 0" means the 141 held-out validation images. "P" is the share of
bootstrap resamples over images in which a change gains PQ.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputConfig:
    """How a grayscale image becomes the 3-channel network input. Identical in training and inference."""
    # The whole disk resized 2048 -> 1536 (INTER_AREA), the size the members were trained at; it fits a 16 GB T4 for
    # training. Inference at other sizes (1408-2048 px, one network, fold 0) never beat it; native 2048 lost ~0.005.
    # Native detail is recovered afterwards by the stage-2 refiner (RefinerConfig).
    input_size: int = 1536
    # Channel 1: image / local background (Gaussian, sigma 64 px) * 0.5. Removes limb darkening and vignetting,
    # so a filament reads the same at disk centre and at the limb.
    background_sigma: float = 64.0
    background_gain: float = 0.5
    # Channel 2: CLAHE (local contrast equalisation).
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    # Channel 3: multi-scale Hessian dark-ridge response, sigma^2-normalised, scaled by its 99.5th percentile.
    # A ridge filter integrates ALONG a thin structure, turning ~1-sigma per-pixel evidence into a strong response.
    ridge_scales: tuple[float, ...] = (2.0, 4.0, 8.0)
    ridge_percentile: float = 99.5


@dataclass(frozen=True)
class InferenceConfig:
    """Test-time augmentation. The 8-view dihedral group (adding the transpose) measured no gain over 4 flips."""
    tta_flips: tuple[tuple[int, ...], ...] = ((), (-1,), (-2,), (-2, -1))   # identity, horizontal, vertical, both


@dataclass(frozen=True)
class DecodeConfig:
    """Probability map -> instances: v24's tuned decode with the fragment gap-merge switched off. Switching it off
    measured +0.0037 PQ [-0.0038, +0.0117] (P 0.82) -- inside seed noise, so not a gain we claim -- and it brought
    merged filaments back from 3.7% to the human rate of 2.8%, with one rule fewer."""
    threshold: float = 0.60     # a pixel is filament if the ensemble probability is >= 0.60
    min_area: int = 240         # native pixels; smaller connected components are dropped
    match_iou: float = 0.50     # the metric's true-positive criterion (IoU > 0.5)


@dataclass(frozen=True)
class FilterConfig:
    """The learned keep/drop filter (HistGradientBoostingRegressor on 15 per-instance features).

    Target = the fraction of an image's annotators who drew a matching filament (IoU > 0.5). Dropping an instance
    with that probability p and matched IoU u changes pooled PQ by -p*u in the numerator and -0.5 in the
    denominator, so dropping helps exactly when p < 0.5 * PQ / SQ: the closed-form threshold."""
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    n_splits: int = 5                     # GroupKFold, grouped by observation date (same-day frames leak)
    seed: int = 0
    grid_offsets: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)   # thresholds tried around the closed form
    near_tie: float = 0.002               # keep the closed form unless a grid value beats it by more than this


@dataclass(frozen=True)
class TrainConfig:
    """One training recipe. The presets in RECIPES are the three shipped members and the teacher-student student."""
    name: str = "v31"
    seed: int = 20260729
    encoder_name: str = "convnext_tiny.dinov3_lvd1689m"     # ConvNeXt-Tiny with DINOv3-distilled weights (timm)
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32)
    input_size: int = 1536
    epochs: int = 30
    lr: float = 3e-4                      # decoder; the pretrained encoder trains at lr * encoder_lr_mult
    encoder_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    batch_size: int = 1                   # a 1536 px image needs ~6 GB under AMP: batch 1 x accumulation 2
    grad_accumulation: int = 2
    grad_clip: float = 1.0
    grad_checkpointing: bool = True       # the fp32 encoder at 1536 px does not fit a 16 GB T4 without it
    num_workers: int = 2
    # Which annotator record is the target: 0 = always the medoid (the record that agrees most with the others),
    # 1 = a random record per sample, in between = a per-sample mix.
    random_annotator_prob: float = 0.5
    # Geometric augmentation: rotation + scale for this share of samples, then the exact D4 group (rot90 + flips).
    affine_prob: float = 0.5
    rotate_deg: float = 15.0
    scale_jitter: float = 0.125
    # GONG-plausible photometric changes (limb darkening, haze, patchy seeing, JPEG), each with this probability.
    photometric_v2_prob: float = 0.0
    # Loss = bce * BCE + dice * [mix * per-instance Dice + (1 - mix) * image Dice] + skeleton * SkeletonRecall
    #        + spine_aux * BCE(spine head).
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    per_instance_dice_mix: float = 0.5
    skeleton_recall_weight: float = 0.5
    spine_aux_weight: float = 0.3
    spine_tube_px: int = 3                # the human-drawn spine polyline rendered 3 px wide (native)
    # Schedule, validation and what ships.
    val_every: int = 3                    # validate every k epochs (and always on the last)
    select_best_epoch: bool = False       # False = ship the final epoch (no selection on the validation images)
    ema_decay: float = 0.0                # exponential moving average of the weights; 0 = off
    max_hours: float | None = None        # stop before an epoch that would not finish inside this budget
    # Teacher-student (see pseudo_labels): warm start, and pseudo-labelled frames drawn per epoch.
    init_from: str | None = None          # a member name (weights/<name>.fp16.pt) to start from
    pseudo_per_epoch: int = 0
    labelled_repeat: int = 1


RECIPES: dict[str, TrainConfig] = {
    # v24: medoid targets, no continuous affine; 35 epochs requested, stopped by the time budget after 23,
    # best validated epoch (17) kept. Fold-0 PQ 0.4494, public LB 0.39.
    "v24": TrainConfig(name="v24", epochs=35, random_annotator_prob=0.0, affine_prob=0.0, val_every=1,
                       select_best_epoch=True, max_hours=8.0),
    # v27: a random annotator record per sample + rotation/scale; 26 epochs, best validated epoch (21) kept.
    # Fold-0 PQ 0.4427, LB 0.38.
    "v27": TrainConfig(name="v27", epochs=26, random_annotator_prob=1.0, select_best_epoch=True),
    # v31: 50/50 medoid/random targets + rotation/scale; 30 epochs, the FINAL epoch ships. Fold-0 PQ 0.4375, LB 0.38.
    "v31": TrainConfig(name="v31", epochs=30, random_annotator_prob=0.5, select_best_epoch=False),
    # Teacher-student: v24 warm start; each epoch = the 566 labelled images + 566 unlabelled GONG
    # frames (2023-2026) drawn with prior-shift weights, soft targets from the 3-member ensemble; lr 1e-4,
    # EMA 0.998, 13 epochs. Fold-0 PQ 0.4487 vs 0.4347 for the same schedule on labelled data only (+0.014, P 1.00).
    # One student matches the 3-model teacher (-0.0009) at a third of the inference cost. LB 0.39.
    "student": TrainConfig(name="student", epochs=13, lr=1e-4, random_annotator_prob=0.0, affine_prob=0.0,
                           photometric_v2_prob=0.3, ema_decay=0.998, val_every=5, init_from="v24",
                           pseudo_per_epoch=566),
}


@dataclass(frozen=True)
class RefinerConfig:
    """Second stage: every kept filament re-segmented at native resolution in its own crop (see refiner.py).
    Fold 0, same 948 kept instances, two seeds: the experiment that adopted it gave PQ 0.4573 -> 0.4690 (+0.0118,
    95% CI [+0.0087, +0.0151]); the shipped pair, retrained from scratch by the main notebook, gives 0.4679 (+0.0106),
    each seed alone 0.4659 / 0.4663. The 0.40 pixel threshold was fixed before the experiment."""
    height: int = 128
    width: int = 384
    along: float = 1.8            # crop length / filament length: the network must see where the filament ends
    margin: int = 24              # native px beyond the filament on every side
    threshold: float = 0.40       # J / (1 + J): the pixel threshold maximising expected IoU for J ~ 0.65
    min_kept: float = 0.5         # a filament keeping less than half its refined area after conflicts is absorbed
    min_area: int = 240           # a refinement smaller than this falls back to the seed (the decode's minimum)
    draw_iou: float = 0.2         # training: an annotator drew a candidate if the best of their filaments has IoU
    draw_cover: float = 0.5       # >= 0.2 with it, or covers >= half of it (the candidate is a truncated piece)
    epochs: int = 16
    batch_size: int = 24
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 0


@dataclass(frozen=True)
class SystemConfig:
    """The deployed system: which members, and how they are run, decoded, filtered and refined."""
    members: tuple[str, ...] = ("v24", "v27", "v31")
    input: InputConfig = field(default_factory=InputConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    refiners: tuple[str, ...] = ("refiner_s0", "refiner_s1")    # weights/<name>.fp16.pt; () switches stage 2 off
    refine: RefinerConfig = field(default_factory=RefinerConfig)
