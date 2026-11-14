"""Storage for probability maps of many images.

Validation maps are kept as uint8 (probability x 255) at native resolution, two channels per image (mask, spine),
in ONE compressed .npz with keys 'p/<stem>' and 's/<stem>'. The 141 fold-0 images take ~11 MB this way (the maps
are mostly zeros), versus 1.1 GB as float arrays, and every consumer reads one image at a time.
"""
from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .inference import dequantise


class MapStore(Mapping):
    """Read-only mapping  filename -> (mask probability, spine probability), float32 (H, W) each."""

    def __init__(self, path: str | Path, suffix: str = ".jpeg") -> None:
        self.path = Path(path)
        with np.load(self.path) as archive:
            stems = sorted(k[2:] for k in archive.files if k.startswith("p/"))
        self._names = {f"{stem}{suffix}": stem for stem in stems}

    def __getitem__(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        stem = self._names[filename]
        with np.load(self.path) as archive:
            return dequantise(archive[f"p/{stem}"]), dequantise(archive[f"s/{stem}"])

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def subset(self, filenames) -> "MapStore":
        """The same store restricted to `filenames` (a quick check on a few images)."""
        store = MapStore.__new__(MapStore)
        store.path, store._names = self.path, {f: self._names[f] for f in filenames}
        return store


def write_map_store(path: str | Path, items: Iterable[tuple[str, np.ndarray]]) -> Path:
    """Stream (filename, uint8 (2, H, W) maps) pairs into a compressed .npz without holding them all in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, maps in items:
            stem = Path(filename).stem
            for key, channel in (("p", maps[0]), ("s", maps[1])):
                with archive.open(f"{key}/{stem}.npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(handle, np.ascontiguousarray(channel, dtype=np.uint8))
    return path
