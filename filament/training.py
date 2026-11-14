"""Training one member (or the teacher-student student).

    fit(...)                   the whole run: data, model, optimiser, epochs, validation, checkpoints, history
      build_loader(...)        labelled images, optionally mixed with pseudo-labelled frames each epoch
      build_optimizer(...)     AdamW; the pretrained encoder at 0.1 x the decoder learning rate
      cosine_factor(...)       learning-rate multiplier: plain cosine from 1 to 0 over the run
      train_one_epoch(...)     AMP, gradient accumulation, clipping, non-finite-loss guard, EMA update
      validate(...)            pooled PQ on validation images (one view, no TTA, the final decode)

The recipes are config.RECIPES. A full 1536 px run takes 13-17 min per epoch on one Tesla T4 (30 epochs ~ 8 h);
the student's epochs are twice as long (566 labelled + 566 pseudo-labelled samples).
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import DecodeConfig, InferenceConfig, InputConfig, TrainConfig
from .data import AnnotationIndex, read_gray
from .datasets import LabelledDataset, MixedEpochSampler
from .decode import decode_instances
from .inference import predict_maps
from .losses import segmentation_loss
from .metrics import aggregate_pq, score_image
from .model import build_model, load_member, save_member

SINGLE_VIEW = InferenceConfig(tta_flips=((),))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_factor(epoch: int, epochs: int) -> float:
    """LR multiplier for 0-based `epoch`: 1 at the start, 0 after the last epoch."""
    return 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    return torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr * cfg.encoder_lr_mult},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)


class EMA:
    """Exponential moving average of the weights, with a bias-corrected warm-up (the average is short early on,
    so the randomly initialised decoder does not dominate it)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        decay = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        ema_params = list(self.model.parameters())
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, [p.detach() for p in model.parameters()], alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)
        self.steps += 1


def build_loader(labelled: Dataset, cfg: TrainConfig, pseudo: Dataset | None = None,
                 pseudo_weights: Sequence[float] | None = None) -> DataLoader:
    """Labelled images shuffled each epoch; with `pseudo`, a MixedEpochSampler adds cfg.pseudo_per_epoch frames."""
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=True, persistent_workers=False)
    if pseudo is None and cfg.labelled_repeat == 1:
        generator = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(labelled, shuffle=True, generator=generator, **common)
    sampler = MixedEpochSampler(len(labelled), cfg.labelled_repeat, len(pseudo) if pseudo is not None else 0,
                                cfg.pseudo_per_epoch, pseudo_weights, seed=cfg.seed)
    return DataLoader(ConcatDataset([labelled] + ([pseudo] if pseudo is not None else [])), sampler=sampler, **common)


def _set_epoch(loader: DataLoader, epoch: int) -> None:
    datasets = loader.dataset.datasets if isinstance(loader.dataset, ConcatDataset) else [loader.dataset]
    for dataset in datasets:
        dataset.epoch = epoch


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, cfg: TrainConfig, device: torch.device, ema: EMA | None = None,
                    progress: Callable | None = None) -> tuple[float, int]:
    """One pass over the loader. Returns (mean loss, number of skipped non-finite batches)."""
    model.train()
    losses, skipped = [], 0
    accumulate = max(cfg.grad_accumulation, 1)
    optimizer.zero_grad(set_to_none=True)
    batches = progress(loader, desc="train", leave=False) if progress else loader
    for step, batch in enumerate(batches, start=1):
        images = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = segmentation_loss(model(images), batch["target"].to(device), batch["spine"].to(device),
                                     batch["components"].to(device), cfg, pseudo=bool(batch["pseudo"].any()))
        if not torch.isfinite(loss):          # never backpropagate a NaN/inf; drop this batch's partial gradients
            skipped += 1
            del loss
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / accumulate).backward()
        if step % accumulate == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        losses.append(loss.item())
    return (float(np.mean(losses)) if losses else float("nan")), skipped


@torch.no_grad()
def validate(model: torch.nn.Module, filenames: Sequence[str], image_dir: str | Path, index: AnnotationIndex,
             device: torch.device, input_cfg: InputConfig = InputConfig(),
             decode_cfg: DecodeConfig = DecodeConfig()) -> dict:
    """Pooled PQ of one model on validation images: single view, the deployed decode, no filter."""
    model.eval()
    rows = []
    for filename in filenames:
        maps = predict_maps([model], read_gray(Path(image_dir) / filename), device, input_cfg, SINGLE_VIEW)
        rows.extend(score_image(filename, decode_instances(maps[0], decode_cfg), index, decode_cfg.match_iou))
    return aggregate_pq(rows)


@dataclass
class TrainResult:
    weights: Path                      # fp16 member file of the model that ships (best or final, EMA if on)
    history: pd.DataFrame              # one row per epoch: loss, lr factor, minutes, validation PQ when run
    meta: dict = field(default_factory=dict)


def fit(cfg: TrainConfig, train_files: Sequence[str], val_files: Sequence[str], image_dir: str | Path,
        index: AnnotationIndex, medoid: Mapping[str, str], out_dir: str | Path, device: torch.device,
        init_weights: str | Path | None = None, pseudo: Dataset | None = None,
        pseudo_weights: Sequence[float] | None = None, log: Callable = print, progress: Callable | None = None,
        on_epoch: Callable[[pd.DataFrame], None] | None = None) -> TrainResult:
    """Train one recipe end to end and write <out_dir>/<cfg.name>.fp16.pt plus history.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    input_cfg = InputConfig(input_size=cfg.input_size)
    loader = build_loader(LabelledDataset(train_files, image_dir, index, medoid, cfg, True, input_cfg), cfg,
                          pseudo, pseudo_weights)
    if init_weights is not None:
        model = load_member(init_weights, device)
        if cfg.grad_checkpointing:
            model.encoder.set_grad_checkpointing(True)
    else:
        model = build_model(cfg.encoder_name, pretrained=True, decoder_channels=cfg.decoder_channels,
                            grad_checkpointing=cfg.grad_checkpointing).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cosine_factor(e, cfg.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=2.0 ** 10)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    torch.backends.cudnn.benchmark = True       # fixed input shape: autotuning pays back within a few steps
    history, best_pq, started = [], -1.0, time.monotonic()
    best_state = None
    for epoch in range(cfg.epochs):
        epoch_start = time.monotonic()
        _set_epoch(loader, epoch)
        loss, skipped = train_one_epoch(model, loader, optimizer, scaler, cfg, device, ema, progress)
        scheduler.step()
        shipped = ema.model if ema is not None else model
        row = {"epoch": epoch + 1, "train_loss": loss, "lr_factor": cosine_factor(epoch, cfg.epochs),
               "skipped_batches": skipped}
        if val_files and ((epoch + 1) % max(cfg.val_every, 1) == 0 or epoch == cfg.epochs - 1):
            val = validate(shipped, val_files, image_dir, index, device, input_cfg)
            row.update({f"val_{k}": v for k, v in val.items() if k in ("pq", "sq", "rq", "tp", "fp", "fn")})
            if cfg.select_best_epoch and val["pq"] > best_pq:
                best_pq = val["pq"]
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in shipped.state_dict().items()}
        row["minutes"] = (time.monotonic() - epoch_start) / 60
        history.append(row)
        frame = pd.DataFrame(history)
        frame.to_csv(out_dir / "history.csv", index=False)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if on_epoch is not None:
            on_epoch(frame)
        elapsed_h = (time.monotonic() - started) / 3600
        if cfg.max_hours and elapsed_h + row["minutes"] / 60 > cfg.max_hours and epoch + 1 < cfg.epochs:
            log(f"stopping after epoch {epoch + 1}: the next epoch would exceed max_hours={cfg.max_hours}")
            break
    final_state = best_state if best_state is not None else (ema.model if ema is not None else model).state_dict()
    meta = {"recipe": asdict(cfg), "epochs_completed": len(history), "selected": "best" if best_state else "final",
            "best_val_pq": best_pq if best_state else None, "hours": round((time.monotonic() - started) / 3600, 2)}
    weights = out_dir / f"{cfg.name}.fp16.pt"
    meta["fingerprint"] = save_member(final_state, weights, meta)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return TrainResult(weights, pd.DataFrame(history), meta)
