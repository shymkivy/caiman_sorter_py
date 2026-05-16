"""Lazy per-cell footprint cache.

Both `ui/image_panel.py` and `ui/nav_panel.py` want the same thing: a dense
`(height, width)` view of a single cell's spatial footprint, keyed by cell
index. Rebuilding it on every UI event (click, arrow-step) is wasteful —
each lookup is an `est.A[:, idx].toarray()` over a 65k-row sparse column
plus a column-major reshape. So we cache, FIFO-evicting once a cap is hit
to bound memory on long sessions.

Used as:
    cache = FootprintCache(maxsize=200)
    ...
    fp_2d = cache.get(est, cell_idx)     # (height, width) float64
    cache.clear()                        # on data_loaded
"""
from __future__ import annotations

import numpy as np


class FootprintCache:
    """FIFO-bounded dense-footprint cache keyed by cell index.

    The cache stores `est.A[:, idx]` reshaped to `(height, width)` in
    column-major (Fortran) order — matching the codebase-wide convention
    that est.A's rows are flattened pixels with `px = row + col * height`.
    """

    def __init__(self, maxsize: int = 200) -> None:
        self._cache: dict[int, np.ndarray] = {}
        self._maxsize = int(maxsize)

    def get(self, est, cell_idx: int) -> np.ndarray:
        """Return the dense `(h, w)` footprint for `cell_idx`, computing on miss."""
        fp = self._cache.get(cell_idx)
        if fp is not None:
            return fp
        fp = np.asarray(est.A[:, cell_idx].toarray()).ravel().reshape(
            est.dims, order="F"
        )
        if len(self._cache) >= self._maxsize:
            # FIFO eviction: drop oldest entry. dict preserves insertion order.
            self._cache.pop(next(iter(self._cache)))
        self._cache[cell_idx] = fp
        return fp

    def clear(self) -> None:
        """Drop all cached entries (call on data_loaded / shape change)."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, cell_idx: int) -> bool:
        return cell_idx in self._cache
