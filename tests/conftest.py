"""Shared fixtures for the caiman_sorter_py test suite.

The fixtures here build small, fully-synthetic Estimates / Proc / Ops
objects so the tests don't depend on any real CaImAn file. Sizes are
small (a few cells, a few thousand frames) so the whole suite runs
in a few seconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from caiman_sorter_py.core.state import (
    DeconvResults, Estimates, Ops, Proc,
)


def _make_footprint(dims: tuple[int, int], cx: int, cy: int,
                    sigma: float = 2.0, amp: float = 1.0) -> np.ndarray:
    """A small 2D Gaussian footprint flattened in column-major (MATLAB) order.

    Matches the convention used everywhere in the codebase:
        px = row + col * height
    so the column-major flatten of (height, width) is what est.A expects.
    """
    h, w = dims
    yy, xx = np.mgrid[0:h, 0:w]
    blob = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    blob[blob < 0.05] = 0.0
    return blob.ravel(order="F")


@pytest.fixture
def tiny_session():
    """Synthetic (Estimates, Proc, Ops) with 5 cells, dims=(16, 16), 1024 frames.

    Cell 0 and cell 1 are placed close together with correlated traces so that
    merge tests can find them as a duplicate pair.
    """
    rng = np.random.default_rng(0)
    dims = (16, 16)
    n_cells = 5
    n_frames = 1024
    fr = 30.0

    # Spatial footprints — cells 0 and 1 overlap; the rest are far apart.
    centers = [(4, 4), (5, 5), (10, 3), (3, 11), (12, 12)]
    A_dense = np.zeros((dims[0] * dims[1], n_cells), dtype=np.float64)
    for i, (cx, cy) in enumerate(centers):
        A_dense[:, i] = _make_footprint(dims, cx, cy, sigma=1.5)
    A = csc_matrix(A_dense)

    # Temporal traces. Cells 0 and 1 share the same underlying signal so their
    # Pearson correlation is high; the rest are independent.
    shared = rng.standard_normal(n_frames) * 0.5 + np.sin(
        np.linspace(0, 20 * np.pi, n_frames)
    )
    C = np.zeros((n_cells, n_frames))
    C[0] = shared + 0.05 * rng.standard_normal(n_frames)
    C[1] = shared + 0.05 * rng.standard_normal(n_frames)
    for i in range(2, n_cells):
        C[i] = rng.standard_normal(n_frames) * 0.3

    YrA = 0.1 * rng.standard_normal((n_cells, n_frames))
    S   = np.maximum(C, 0) * 0.5
    F_dff = C + YrA

    # Background — 2-rank spatial + temporal background (typical CaImAn shapes).
    n_bg = 2
    b = rng.standard_normal((dims[0] * dims[1], n_bg)).astype(np.float64) * 0.01
    f = rng.standard_normal((n_bg, n_frames)).astype(np.float64) * 0.5

    est = Estimates(
        A=A,
        C=C,
        YrA=YrA,
        S=S,
        F_dff=F_dff,
        SNR_comp=np.array([5.0, 4.5, 6.0, 3.5, 4.0]),
        cnn_preds=np.array([0.99, 0.98, 0.9, 0.5, 0.8]),
        r_values=np.array([0.9, 0.85, 0.95, 0.6, 0.7]),
        g=np.tile(np.array([[0.95], [-0.1]]), (1, n_cells)),
        dims=dims,
        idx_components=np.arange(n_cells, dtype=np.int64),
        idx_components_bad=np.array([], dtype=np.int64),
        sn=rng.uniform(0.05, 0.1, dims[0] * dims[1]).astype(np.float64),
        neurons_sn=rng.uniform(0.05, 0.1, n_cells).astype(np.float64),
        b=b,
        f=f,
        init_params_caiman={
            # Nested layout (matches CaImAn HDF5 /params/ tree).
            "data":       {"fr": float(fr), "dims": list(dims)},
            "preprocess": {"p": 2},
            "temporal":   {"fudge_factor": 0.97, "lags": 5},
            "init":       {"gSig": [3, 3]},
        },
        eval_params_caiman={"SNR_thresh": 2.0},
        num_cells_original=n_cells,
    )

    accepted = np.array([True, True, True, False, True])
    proc = Proc(
        num_cells=n_cells,
        num_frames=n_frames,
        accepted=accepted.copy(),
        accepted_core=accepted.copy(),
        manual_override=np.zeros(n_cells, dtype=bool),
        noise=rng.uniform(0.05, 0.1, n_cells),
        skewness=rng.standard_normal(n_cells),
        peaks_ave=rng.uniform(0.5, 2.0, n_cells),
        num_zeros=rng.integers(0, n_frames, n_cells).astype(np.float64),
        SNR2_vals=np.array([10.0, 8.0, 12.0, 2.0, 5.0]),  # cell 0 wins vs 1 for merge tests
        firing_stab_vals=rng.uniform(0.0, 1.0, n_cells),
        gAR1=rng.uniform(0.9, 0.99, n_cells),
        gAR2=rng.uniform(0.5, 0.9, (n_cells, 2)),
        tauAR1=rng.uniform(0.3, 0.6, n_cells),
        tauAR2=rng.uniform(0.05, 0.5, (n_cells, 2)),
        smooth_dfdt=DeconvResults(
            S=[None] * n_cells, C=[None] * n_cells, g=[None] * n_cells,
        ),
        foopsi=DeconvResults(
            S=[None] * n_cells, C=[None] * n_cells, g=[None] * n_cells,
        ),
        smooth_dfdt_std=rng.uniform(0.01, 0.1, n_cells),
    )

    # Mark cell 2 as having a foopsi result so we exercise the populated-row path.
    proc.foopsi.S[2] = rng.standard_normal(n_frames)
    proc.foopsi.C[2] = rng.standard_normal(n_frames)
    proc.foopsi.g[2] = np.array([0.94, -0.08])

    ops = Ops()
    ops.foopsi.fudge_factor = 0.97
    ops.foopsi.ar_order     = 2

    return est, proc, ops
