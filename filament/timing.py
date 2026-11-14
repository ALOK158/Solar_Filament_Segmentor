"""Wall-clock accounting per pipeline stage (read, preprocess, network, upsample, decode, filter, refine, encode)."""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict

import pandas as pd
import torch


class StageTimer:
    """`with timer("network"): ...` adds the block's wall time to that stage. GPU work is synchronised before and
    after a stage marked `sync=True`, so asynchronous CUDA kernels are charged to the stage that launched them."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def __call__(self, stage: str, sync: bool = False):
        if sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.seconds[stage] += time.perf_counter() - start
            self.calls[stage] += 1

    def table(self, images: int | None = None) -> pd.DataFrame:
        total = sum(self.seconds.values())
        frame = pd.DataFrame({"seconds": pd.Series(self.seconds), "calls": pd.Series(self.calls)})
        frame["share"] = frame["seconds"] / max(total, 1e-9)
        if images:
            frame["s_per_image"] = frame["seconds"] / images
        return frame.sort_values("seconds", ascending=False).round(3)
