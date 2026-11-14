"""Second stage: re-segment every kept filament at native resolution, in its own crop.

The dense network sees the whole disk at 1536 px, so every instance is a level set of an upsampled 1536-grid map:
edges can sit a pixel off, and a filament whose probability dips part-way is cut short (truncation is the most
common near miss). The refiner looks at one kept filament at a time at native 2048 px:
  crop    rotated so the filament's long axis is horizontal, `along` x its length (the network must see where the
          filament ends) and `margin` px either side; shrunk only if it does not fit 128 x 384, never enlarged.
  input   7 channels: the 3 image channels (preprocess.to_channels at native resolution), the ensemble's mask and
          spine probabilities, the filament's own mask (the seed) and the other kept filaments (where a neighbour
          begins).
  target  per pixel, the share of the annotators who drew THIS filament that mark the pixel (whether the filament
          exists is the instance filter's decision, so annotators who did not draw it are left out).
  output  thresholded at 0.40 = J / (1 + J), the pixel threshold that maximises expected IoU for J ~ 0.65.
Paste-back keeps the refined components that touch the seed; a pixel claimed by two filaments goes to the higher
probability; a filament left with under half of its refined area is absorbed; a failed refinement keeps the seed.
Instances stay disjoint, as the host requires.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import RefinerConfig
from .metrics import pairwise_iou
from .model import build_model, read_member

CHANNELS = ("background", "clahe", "ridge", "mask_probability", "spine_probability", "seed", "others")


# ----------------------------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------------------------

def crop_affine(mask: np.ndarray, cfg: RefinerConfig = RefinerConfig()) -> np.ndarray:
    """2 x 3 affine (image x, y -> crop u, v) that lays the filament's principal axis along the crop's width."""
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centre = points.mean(axis=0)
    major = np.linalg.eigh(np.cov((points - centre).T))[1][:, 1] if len(points) > 2 else np.array([1.0, 0.0])
    major = major if major[0] >= 0 else -major
    minor = np.array([-major[1], major[0]])
    a, b = (points - centre) @ major, (points - centre) @ minor
    mid = centre + major * (a.min() + a.max()) / 2 + minor * (b.min() + b.max()) / 2
    length, width = a.max() - a.min() + 1, b.max() - b.min() + 1
    scale = min(1.0, cfg.width / (length * cfg.along + 2 * cfg.margin), cfg.height / (width + 2 * cfg.margin))
    affine = np.zeros((2, 3))
    affine[0, :2], affine[1, :2] = scale * major, scale * minor
    affine[0, 2] = cfg.width / 2 - scale * (mid @ major)
    affine[1, 2] = cfg.height / 2 - scale * (mid @ minor)
    return affine


def warp(channels: Sequence[np.ndarray], affine: np.ndarray, cfg: RefinerConfig = RefinerConfig()) -> np.ndarray:
    """Full-resolution channels -> (C, height, width) crop, bilinear, zero outside the image."""
    return np.stack([cv2.warpAffine(np.asarray(c, np.float32), affine, (cfg.width, cfg.height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                     for c in channels])


def unwarp(crop: np.ndarray, affine: np.ndarray, shape: tuple[int, int],
           cfg: RefinerConfig = RefinerConfig()) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """(height, width) crop -> ((y0, y1, x0, x1), the crop resampled onto that box of the full image)."""
    inverse = cv2.invertAffineTransform(affine)
    corners = np.array([[0, 0], [cfg.width, 0], [0, cfg.height], [cfg.width, cfg.height]], np.float64)
    full = corners @ inverse[:, :2].T + inverse[:, 2]
    x0, x1 = max(int(math.floor(full[:, 0].min())) - 1, 0), min(int(math.ceil(full[:, 0].max())) + 2, shape[1])
    y0, y1 = max(int(math.floor(full[:, 1].min())) - 1, 0), min(int(math.ceil(full[:, 1].max())) + 2, shape[0])
    shifted = affine.copy()
    shifted[:, 2] += affine[:, :2] @ np.array([x0, y0], np.float64)
    region = cv2.warpAffine(np.asarray(crop, np.float32), shifted, (x1 - x0, y1 - y0),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    return (y0, y1, x0, x1), region


# ----------------------------------------------------------------------------------------------------------------
# Crops
# ----------------------------------------------------------------------------------------------------------------

def instance_crops(channels: np.ndarray, probability: np.ndarray, spine: np.ndarray,
                   instances: Sequence[np.ndarray], cfg: RefinerConfig = RefinerConfig()
                   ) -> tuple[np.ndarray, list[np.ndarray]]:
    """(N, 7, height, width) float32 crops of disjoint instances + their affines. `channels` = to_channels(image)
    at native resolution; each crop's 'others' channel is the union of the other instances."""
    union = np.zeros(probability.shape, np.float32)
    for mask in instances:
        union[mask] = 1.0
    crops, affines = [], []
    for mask in instances:
        affine = crop_affine(mask, cfg)
        seed = mask.astype(np.float32)
        crops.append(warp([*channels, probability, spine, seed, union - seed], affine, cfg))
        affines.append(affine)
    if not crops:
        return np.zeros((0, len(CHANNELS), cfg.height, cfg.width), np.float32), []
    return np.stack(crops), affines


def consensus_targets(candidates: Sequence[np.ndarray], gt_sets: Sequence[Sequence[np.ndarray]],
                      cfg: RefinerConfig = RefinerConfig()) -> list[np.ndarray | None]:
    """Per candidate: the mean of the filaments that the annotators who drew it drew (float32, full size), or None
    when nobody drew it. An annotator draws a candidate if one of their filaments overlaps it enough (config)."""
    areas = np.array([m.sum() for m in candidates], np.float64)
    sums: list[np.ndarray | None] = [None] * len(candidates)
    counts = np.zeros(len(candidates))
    for gt in gt_sets:
        if not gt or not len(candidates):
            continue
        iou = pairwise_iou(list(gt), list(candidates))                     # (n_gt, n_candidates)
        gt_areas = np.array([g.sum() for g in gt], np.float64)
        best = iou.argmax(axis=0)
        for k, j in enumerate(best):
            value = iou[j, k]
            cover = value * (gt_areas[j] + areas[k]) / (1.0 + value) / max(areas[k], 1.0)   # |g n c| / |c|
            if value >= cfg.draw_iou or cover >= cfg.draw_cover:
                sums[k] = gt[j].astype(np.float32) if sums[k] is None else sums[k] + gt[j]
                counts[k] += 1
    return [s / c if s is not None else None for s, c in zip(sums, counts)]


def training_examples(channels: np.ndarray, probability: np.ndarray, spine: np.ndarray,
                      candidates: Sequence[np.ndarray], gt_sets: Sequence[Sequence[np.ndarray]],
                      cfg: RefinerConfig = RefinerConfig()) -> tuple[np.ndarray, np.ndarray]:
    """uint8 crops (N, 7, h, w) and uint8 consensus targets (N, h, w) of the candidates somebody drew."""
    targets = consensus_targets(candidates, gt_sets, cfg)
    crops, affines = instance_crops(channels, probability, spine, candidates, cfg)
    keep = [k for k, t in enumerate(targets) if t is not None]
    if not keep:
        return (np.zeros((0, len(CHANNELS), cfg.height, cfg.width), np.uint8),
                np.zeros((0, cfg.height, cfg.width), np.uint8))
    x = np.round(np.clip(crops[keep], 0, 1) * 255).astype(np.uint8)
    y = np.stack([np.round(np.clip(warp([targets[k]], affines[k], cfg)[0], 0, 1) * 255).astype(np.uint8)
                  for k in keep])
    return x, y


def image_examples(probability: np.ndarray, spine: np.ndarray, channels: np.ndarray, gt_sets,
                   cfg: RefinerConfig = RefinerConfig(), levels: Sequence[float] = (0.5, 0.6, 0.7)
                   ) -> tuple[np.ndarray, np.ndarray]:
    """uint8 training crops and consensus targets of ONE training image (see training_set)."""
    from .config import DecodeConfig
    from .decode import decode_instances

    xs, ys, taken = [], [], []
    for level in levels:
        candidates = decode_instances(probability, DecodeConfig(threshold=level))
        if not candidates:
            continue
        new = [k for k, c in enumerate(candidates) if not taken or pairwise_iou(taken, [c]).max() < 0.98]
        taken.extend(candidates[k] for k in new)
        targets = consensus_targets(candidates, gt_sets, cfg)
        new = [k for k in new if targets[k] is not None]
        if not new:
            continue
        crops, affines = instance_crops(channels, probability, spine, candidates, cfg)
        xs.append(np.round(np.clip(crops[new], 0, 1) * 255).astype(np.uint8))
        ys.append(np.stack([np.round(np.clip(warp([targets[k]], affines[k], cfg)[0], 0, 1) * 255).astype(np.uint8)
                            for k in new]))
    if not xs:
        return (np.zeros((0, len(CHANNELS), cfg.height, cfg.width), np.uint8),
                np.zeros((0, cfg.height, cfg.width), np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


_JOB: dict = {}       # per-process state of the crop workers (set once by _init_crop_worker)


def _init_crop_worker(maps, image_dir, index, cfg, levels) -> None:
    import cv2 as _cv2

    _cv2.setNumThreads(1)
    _JOB.update(maps=maps, image_dir=Path(image_dir), index=index, cfg=cfg, levels=levels)


def _crop_worker(filename: str):
    from .data import ground_truth_sets, read_gray
    from .preprocess import to_channels

    probability, spine = _JOB["maps"][filename]
    channels = to_channels(read_gray(_JOB["image_dir"] / filename))
    return image_examples(probability, spine, channels, ground_truth_sets(_JOB["index"], filename), _JOB["cfg"],
                          _JOB["levels"])


def training_set(maps, filenames: Sequence[str], image_dir: str | Path, index, cfg: RefinerConfig = RefinerConfig(),
                 levels: Sequence[float] = (0.5, 0.6, 0.7), progress: Callable | None = None, workers: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Training crops from the ensemble's maps of the TRAINING images: candidates at several decode thresholds
    (wider and narrower extents than the deployed 0.60), a candidate repeated across levels (IoU >= 0.98 with one
    already taken) kept once, and only candidates some annotator drew. Each level's 'others' channel is that level's
    other candidates. The shipped recipe: levels 0.5 / 0.6 / 0.7 -> 12,683 crops from 561 of the 566 images.
    `workers` > 1 builds the images in parallel processes (same crops, same order); `maps` must then be picklable,
    as a MapStore is."""
    filenames = list(filenames)
    _init_crop_worker(maps, image_dir, index, cfg, levels)
    if workers > 1:
        import multiprocessing as mp

        context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with context.Pool(workers, _init_crop_worker, (maps, image_dir, index, cfg, levels)) as pool:
            results = pool.imap(_crop_worker, filenames, chunksize=2)
            results = list(progress(results, desc="refiner crops", total=len(filenames)) if progress else results)
    else:
        items = progress(filenames, desc="refiner crops") if progress else filenames
        results = [_crop_worker(f) for f in items]
    results = [r for r in results if len(r[0])]
    if not results:
        return (np.zeros((0, len(CHANNELS), cfg.height, cfg.width), np.uint8),
                np.zeros((0, cfg.height, cfg.width), np.uint8))
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


# ----------------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------------

def build_refiner(pretrained: bool = False):
    return build_model(pretrained=pretrained, input_channels=len(CHANNELS), heads=1)


def load_refiner(path, device: torch.device | str = "cpu"):
    state, _ = read_member(path)
    model = build_refiner()
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if set(missing) - {"mean", "std"} or unexpected:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def predict_crops(models: Sequence[torch.nn.Module], crops: np.ndarray, device: torch.device | str,
                  batch_size: int = 32, flips: Sequence[tuple[int, ...]] = ((), (2,), (3,), (2, 3))) -> np.ndarray:
    """(N, 7, h, w) crops in [0, 1] -> (N, h, w) probability, mean over models x flips."""
    out = np.zeros((len(crops), crops.shape[-2], crops.shape[-1]), np.float32)
    for start in range(0, len(crops), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(crops[start:start + batch_size], np.float32)).to(device)
        total = torch.zeros(x.shape[0], 1, *x.shape[-2:], device=device)
        for model in models:
            for dims in flips:
                y = model(x.flip(dims) if dims else x)[:, :1].float().sigmoid()
                total += y.flip(dims) if dims else y
        out[start:start + len(x)] = (total[:, 0] / (len(models) * len(flips))).cpu().numpy()
    return out


# ----------------------------------------------------------------------------------------------------------------
# Paste-back
# ----------------------------------------------------------------------------------------------------------------

def paste_back(instances: Sequence[np.ndarray], crop_probabilities: np.ndarray, affines: Sequence[np.ndarray],
               cfg: RefinerConfig = RefinerConfig(), return_index: bool = False):
    """Refined crops -> disjoint full-size instance masks (see the module docstring for the rules). With
    `return_index`, also the positions in `instances` of the masks that survive (absorbed ones drop out)."""
    if not len(instances):
        return ([], []) if return_index else []
    shape = instances[0].shape
    best = np.zeros(shape, np.float32)
    owner = np.full(shape, -1, np.int32)
    regions = []
    for k, (seed_full, crop, affine) in enumerate(zip(instances, crop_probabilities, affines)):
        (y0, y1, x0, x1), region = unwarp(crop, affine, shape, cfg)
        seed = seed_full[y0:y1, x0:x1]
        foreground = region >= cfg.threshold
        _, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
        touching = np.unique(labels[seed & foreground])
        foreground = np.isin(labels, touching[touching > 0])
        if foreground.sum() < cfg.min_area:                       # refinement failed: keep the seed
            foreground = seed.copy()
            region = np.where(seed, np.maximum(region, cfg.threshold), region)
        regions.append(((y0, y1, x0, x1), foreground))
        sub_best, sub_owner = best[y0:y1, x0:x1], owner[y0:y1, x0:x1]
        wins = foreground & (region > sub_best)
        sub_best[wins], sub_owner[wins] = region[wins], k
    refined, index = [], []
    for k, ((y0, y1, x0, x1), foreground) in enumerate(regions):
        mine = foreground & (owner[y0:y1, x0:x1] == k)
        if not mine.any() or mine.sum() < cfg.min_kept * foreground.sum():
            continue                                                # absorbed by a neighbour
        mask = np.zeros(shape, bool)
        mask[y0:y1, x0:x1] = mine
        refined.append(mask)
        index.append(k)
    return (refined, index) if return_index else refined


def refine(models: Sequence[torch.nn.Module], channels: np.ndarray, probability: np.ndarray, spine: np.ndarray,
           instances: Sequence[np.ndarray], device: torch.device | str,
           cfg: RefinerConfig = RefinerConfig()) -> tuple[list[np.ndarray], list[int]]:
    """Kept instances of one image -> (refined disjoint instances, their positions in `instances`). Crops are
    rounded to 8 bits first, exactly as the training crops were stored."""
    crops, affines = instance_crops(channels, probability, spine, instances, cfg)
    if not len(crops):
        return [], []
    crops = np.round(np.clip(crops, 0, 1) * 255) / 255.0
    return paste_back(instances, predict_crops(models, crops, device), affines, cfg, return_index=True)


def load_refiners(weights_dir: str | Path, names: Sequence[str], device: torch.device | str = "cpu") -> list:
    return [load_refiner(Path(weights_dir) / f"{name}.fp16.pt", device) for name in names]


# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------

def _seed_worker(worker_id: int, base: int) -> None:
    """DataLoader worker seed (a top-level function, so it pickles under any start method)."""
    np.random.seed(base * 100 + worker_id)


class CropDataset(torch.utils.data.Dataset):
    """uint8 crops + targets (memory-mapped arrays are fine). Training jitter: flips along both crop axes, image
    brightness x U(0.9, 1.1), and the seed eroded or dilated by one pixel half of the time (the refiner must not
    simply copy the seed's edge)."""

    def __init__(self, crops: np.ndarray, targets: np.ndarray, train: bool = True, seed: int = 0) -> None:
        self.crops, self.targets, self.train = crops, targets, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int):
        x = self.crops[i].astype(np.float32) / 255.0
        y = self.targets[i].astype(np.float32)[None] / 255.0
        if self.train:
            rng = self.rng
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if rng.random() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
            x = x.copy()
            x[:3] = np.clip(x[:3] * rng.uniform(0.9, 1.1), 0, 1)
            if rng.random() < 0.5:
                seed = (x[5] >= 0.5).astype(np.uint8)
                op = cv2.dilate if rng.random() < 0.5 else cv2.erode
                x[5] = op(seed, np.ones((3, 3), np.uint8)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def refiner_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft-target BCE + per-crop soft Dice (Dice keeps thin ribbons from collapsing to background)."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1.0) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)
    return bce + dice.mean()


def fit_refiner(crops: np.ndarray, targets: np.ndarray, device: torch.device | str,
                cfg: RefinerConfig = RefinerConfig(), pretrained: bool = True, log: Callable = print,
                num_workers: int = 2):
    """Train one refiner (AdamW, 1-epoch warm-up then cosine, AMP outside the fp32 encoder). Returns
    (model, history rows)."""
    torch.manual_seed(cfg.seed)
    model = build_refiner(pretrained).to(device)
    import functools
    import multiprocessing as mp

    # Loader workers fork on Linux: nothing is pickled, and the memory-mapped crops are shared, not copied. This
    # matters inside fit_refiners' spawned processes, whose default start method is spawn.
    context = mp.get_context("fork") if num_workers and "fork" in mp.get_all_start_methods() else None
    loader = torch.utils.data.DataLoader(CropDataset(crops, targets, True, cfg.seed), batch_size=cfg.batch_size,
                                         shuffle=True, drop_last=True, num_workers=num_workers,
                                         worker_init_fn=functools.partial(_seed_worker, base=cfg.seed),
                                         multiprocessing_context=context)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps, warmup = cfg.epochs * len(loader), len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))))
    use_amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
                loss = refiner_loss(model(x), y)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total, seen = total + float(loss) * len(x), seen + len(x)
        history.append({"epoch": epoch + 1, "loss": total / max(seen, 1), "lr": optimiser.param_groups[0]["lr"]})
        log(f"refiner seed {cfg.seed} epoch {epoch + 1}/{cfg.epochs} loss {history[-1]['loss']:.4f}")
    return model.eval(), history


def _fit_job(job: dict) -> None:
    """One refiner in its own process: memory-mapped crops, fp16 weight file + history JSON written to disk."""
    import json
    import time
    from dataclasses import replace

    from .model import save_member

    start = time.time()
    crops = np.load(job["crops"], mmap_mode="r")
    targets = np.load(job["targets"], mmap_mode="r")
    cfg = replace(job["cfg"], seed=job["seed"])
    net, history = fit_refiner(crops, targets, job["device"], cfg, job["pretrained"],
                               log=lambda m: print(f"[{job['name']} on {job['device']}] {m}", flush=True),
                               num_workers=job["num_workers"])
    fingerprint = save_member(net, job["out"], {"name": job["name"], "kind": "refiner", "seed": job["seed"],
                                               "channels": list(CHANNELS), "crops": len(crops)})
    Path(job["out"]).with_suffix(".json").write_text(json.dumps(
        {"fingerprint": fingerprint, "history": history, "minutes": (time.time() - start) / 60}))


def fit_refiners(crops: np.ndarray, targets: np.ndarray, names: Sequence[str], devices: Sequence[str],
                 out_dir: str | Path, cfg: RefinerConfig = RefinerConfig(), pretrained: bool = True,
                 num_workers: int = 2) -> dict[str, dict]:
    """Train one refiner per name; 'refiner_s1' trains with seed 1. With several devices the refiners train at the
    same time, one process per device (the shipped pair: two T4s, 82 min instead of ~2.7 h); otherwise in turn.
    Returns name -> {path, fingerprint, history, minutes}."""
    import json
    import multiprocessing as mp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "crops.npy", crops)
    np.save(out_dir / "targets.npy", targets)
    jobs = [{"name": name, "seed": int(name.rsplit("s", 1)[-1]), "device": str(devices[i % len(devices)]),
             "crops": str(out_dir / "crops.npy"), "targets": str(out_dir / "targets.npy"), "cfg": cfg,
             "pretrained": pretrained, "num_workers": num_workers, "out": str(out_dir / f"{name}.fp16.pt")}
            for i, name in enumerate(names)]
    if len(set(job["device"] for job in jobs)) > 1:
        if pretrained:
            build_refiner(pretrained=True)        # fetch the encoder weights once, before two processes race for them
        context = mp.get_context("spawn")
        processes = [context.Process(target=_fit_job, args=(job,)) for job in jobs]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [job["name"] for job, process in zip(jobs, processes) if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"refiner training failed: {failed}")
    else:
        for job in jobs:
            _fit_job(job)
    for array in ("crops.npy", "targets.npy"):            # ~5 GB for the full set: not kept
        (out_dir / array).unlink(missing_ok=True)
    results = {}
    for job in jobs:
        info = json.loads(Path(job["out"]).with_suffix(".json").read_text())
        results[job["name"]] = {"path": job["out"], **info}
    return results
