"""Display-only orientation transforms (rotation + flips).

A `PlotParams` value defines a 90°-step rotation plus optional horizontal /
vertical flips. The transform is applied at render time in the image and nav
panels — it never mutates `est.A`, `proc`, contours, or the saved file format.

Convention:
  - Image arrays are shape `(H, W)` (rows × cols).
  - With `imshow(..., origin="upper")`, `arr[0, 0]` is the top-left pixel;
    increasing row goes DOWN, increasing col goes RIGHT.
  - Spatial coordinates are `(x, y) = (col, row)`.
  - Order of operations: flip_h, then flip_v, then rotate `rotation` degrees CCW.

Provides:
  - displayed_dims(dims, p)   : new (H, W) after transform
  - transform_image(arr, p)   : transformed 2D array
  - transform_xy(xy, dims, p) : transformed (N, 2) coord array + new dims
  - inverse_xy(x, y, dims, p) : displayed (x, y) → original (x, y) scalar
"""
from __future__ import annotations

import numpy as np


def _k(rotation: int) -> int:
    """Return the rotation expressed as the integer arg to np.rot90 (0..3)."""
    return (int(rotation) // 90) % 4


def displayed_dims(dims: tuple[int, int], plot_params) -> tuple[int, int]:
    """Return (H, W) after applying the rotation. Flips don't change shape."""
    H, W = int(dims[0]), int(dims[1])
    return (W, H) if _k(plot_params.rotation) % 2 else (H, W)


def transform_image(arr: np.ndarray, plot_params) -> np.ndarray:
    """Apply flip_h, flip_v, then rotation to a 2D array."""
    if plot_params.flip_h:
        arr = arr[:, ::-1]
    if plot_params.flip_v:
        arr = arr[::-1, :]
    k = _k(plot_params.rotation)
    if k:
        arr = np.rot90(arr, k=k)
    return np.ascontiguousarray(arr)


def transform_xy(xy: np.ndarray, dims: tuple[int, int],
                 plot_params) -> tuple[np.ndarray, tuple[int, int]]:
    """Transform an (N, 2) array of (x, y) coords.

    `dims` is the ORIGINAL `(H, W)`. Returns the transformed coords plus the
    `(H, W)` of the displayed image after rotation.
    """
    if xy is None or len(xy) == 0:
        return xy, displayed_dims(dims, plot_params)
    H, W = int(dims[0]), int(dims[1])
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    if plot_params.flip_h:
        x = (W - 1) - x
    if plot_params.flip_v:
        y = (H - 1) - y
    k = _k(plot_params.rotation)
    if k == 0:
        out_x, out_y, new_dims = x, y, (H, W)
    elif k == 1:
        out_x, out_y, new_dims = y, (W - 1) - x, (W, H)
    elif k == 2:
        out_x, out_y, new_dims = (W - 1) - x, (H - 1) - y, (H, W)
    else:
        out_x, out_y, new_dims = (H - 1) - y, x, (W, H)
    return np.column_stack([out_x, out_y]), new_dims


def inverse_xy(x: float, y: float, dims: tuple[int, int],
               plot_params) -> tuple[float, float]:
    """Map a displayed (x, y) back to the original (x, y).

    `dims` is the ORIGINAL `(H, W)` — i.e. `est.dims`, NOT the displayed shape.
    """
    H, W = int(dims[0]), int(dims[1])
    k = _k(plot_params.rotation)
    if k == 0:
        x_post, y_post = x, y
    elif k == 1:
        x_post, y_post = (W - 1) - y, x
    elif k == 2:
        x_post, y_post = (W - 1) - x, (H - 1) - y
    else:
        x_post, y_post = y, (H - 1) - x
    if plot_params.flip_h:
        x_post = (W - 1) - x_post
    if plot_params.flip_v:
        y_post = (H - 1) - y_post
    return x_post, y_post
