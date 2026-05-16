"""Contour computation from spatial components.

Uses skimage.measure.find_contours to trace the outer boundary of each
cell's thresholded footprint at subpixel precision, rather than CaImAn's
energy-based method which draws contours inside the bright core.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix


def compute_contours(A: csc_matrix, dims: tuple[int, int],
                     thr: float = 0.01) -> list:
    """Compute outer-boundary contours for each spatial component.

    Thresholds each cell's footprint at `thr * max_value`, then traces
    the boundary of that binary mask using skimage.measure.find_contours.
    This gives a contour that fully outlines the cell at pixel boundaries.

    Args:
        A:    (n_pixels, n_cells) sparse spatial components matrix.
        dims: (height, width) of the imaging FOV.
        thr:  Amplitude threshold as fraction of each cell's peak value.
              Lower values include more of the footprint (default 0.01).

    Returns:
        List of dicts, one per cell:
          'coordinates': (N, 2) float array of (x, y) = (col, row) points,
                         or None if no contour found.
          'CoM':         [row, col] weighted centre of mass, or None.
    """
    try:
        from skimage.measure import find_contours
    except ImportError:
        return _fallback_caiman(A, dims)

    height, width = dims
    A = A.tocsc()
    indptr  = A.indptr
    indices = A.indices
    data    = A.data
    n_cells = A.shape[1]
    result  = []

    for i in range(n_cells):
        start, end = int(indptr[i]), int(indptr[i + 1])
        if start == end:
            result.append({"coordinates": None, "CoM": None})
            continue

        nz_px = indices[start:end]
        vals  = data[start:end]
        # Linear pixel index → (row, col) via column-major layout (Fortran order):
        #   px = row + col * height  =>  row = px % height,  col = px // height
        nz_rows = nz_px %  height
        nz_cols = nz_px // height

        fp_max = float(vals.max())
        if fp_max == 0:
            result.append({"coordinates": None, "CoM": None})
            continue

        # Threshold and select kept pixels
        thr_abs   = thr * fp_max
        keep_mask = vals > thr_abs
        if not keep_mask.any():
            result.append({"coordinates": None, "CoM": None})
            continue
        kept_rows = nz_rows[keep_mask]
        kept_cols = nz_cols[keep_mask]
        kept_vals = vals[keep_mask]

        # Weighted centre of mass on the kept pixels
        com = [float(np.average(kept_rows, weights=kept_vals)),
               float(np.average(kept_cols, weights=kept_vals))]

        # Allocate a bounding-box buffer (+1 pad on each side so find_contours
        # at level=0.5 traces a closed loop even when the footprint touches
        # the FOV edge). Clip the global bbox to [0, dims).
        pad = 1
        r0 = max(0, int(kept_rows.min()) - pad)
        r1 = min(height - 1, int(kept_rows.max()) + pad)
        c0 = max(0, int(kept_cols.min()) - pad)
        c1 = min(width  - 1, int(kept_cols.max()) + pad)
        box = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float32)
        box[kept_rows - r0, kept_cols - c0] = 1.0

        segs = find_contours(box, level=0.5)
        if not segs:
            result.append({"coordinates": None, "CoM": com})
            continue
        seg = max(segs, key=len)
        # Offset back to global coords. skimage returns (row, col); we emit
        # (x=col, y=row) for imshow.
        coords_xy = np.column_stack([seg[:, 1] + c0, seg[:, 0] + r0])
        result.append({"coordinates": coords_xy, "CoM": com})

    return result


def _fallback_caiman(A: csc_matrix, dims: tuple[int, int]) -> list | None:
    """Fall back to CaImAn's get_contours if skimage is unavailable."""
    try:
        from caiman.utils.visualization import get_contours
        return get_contours(A, dims, thr=0.2)
    except Exception:
        return None
