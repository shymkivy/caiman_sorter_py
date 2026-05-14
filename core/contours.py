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

    n_cells = A.shape[1]
    result  = []

    for i in range(n_cells):
        fp = np.asarray(A[:, i].todense()).ravel().reshape(dims, order="F")
        fp_max = float(fp.max())

        if fp_max == 0:
            result.append({"coordinates": None, "CoM": None})
            continue

        binary = (fp > thr * fp_max).astype(np.float32)

        # Weighted centre of mass
        rows, cols = np.where(binary)
        if len(rows):
            w   = fp[rows, cols]
            com = [float(np.average(rows, weights=w)),
                   float(np.average(cols, weights=w))]
        else:
            com = None

        # Find outer contour of the binary mask (boundary between 0 and 1)
        segs = find_contours(binary, level=0.5)
        if not segs:
            result.append({"coordinates": None, "CoM": com})
            continue

        # Take the longest segment (outer boundary, not holes)
        seg = max(segs, key=len)
        # skimage returns (row, col); convert to (x, y) = (col, row) for imshow
        coords_xy = np.column_stack([seg[:, 1], seg[:, 0]])

        result.append({"coordinates": coords_xy, "CoM": com})

    return result


def _fallback_caiman(A: csc_matrix, dims: tuple[int, int]) -> list | None:
    """Fall back to CaImAn's get_contours if skimage is unavailable."""
    try:
        from caiman.utils.visualization import get_contours
        return get_contours(A, dims, thr=0.2)
    except Exception:
        return None
