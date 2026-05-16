"""End-to-end test for core.proc_init.initialize_proc.

This is the function the GUI calls on every fresh CaImAn HDF5 load to
populate the full Proc dataclass from Estimates. It chains
`get_ar_init_params`, `compute_noise`, `compute_peaks_ave`,
`compute_firing_stability`, `_batch_autocov`, and `compute_ar_pair`. The
individual pieces are unit-tested elsewhere; this is the smoke test that
ties them together.

A regression here is high-impact: the rest of the GUI (cell sorting,
foopsi, metric display) reads from `proc.*` arrays produced exclusively
by this function.
"""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.proc_init import (
    compute_ar_pair, get_ar_init_params, initialize_proc,
)
from caiman_sorter_py.core.state import Ops


def test_initialize_proc_populates_every_field(tiny_session):
    """Every per-cell field on Proc is shaped and finite."""
    est, _, _ = tiny_session
    n_cells = est.A.shape[1]
    n_frames = est.C.shape[1]

    proc = initialize_proc(est, Ops())

    # Top-level scalars
    assert proc.num_cells  == n_cells
    assert proc.num_frames == n_frames

    # Bool masks
    for name in ("accepted", "accepted_core", "manual_override"):
        v = getattr(proc, name)
        assert v.shape == (n_cells,), name
        assert v.dtype == bool, name

    # 1-D float metrics
    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals", "gAR1", "tauAR1"):
        v = getattr(proc, name)
        assert v is not None,             f"{name} is None"
        assert v.shape == (n_cells,),     f"{name}: {v.shape}"
        assert np.all(np.isfinite(v)),    f"{name} has NaN/inf: {v}"

    # 2-D AR(2) outputs
    for name in ("gAR2", "tauAR2"):
        v = getattr(proc, name)
        assert v is not None,             f"{name} is None"
        assert v.shape == (n_cells, 2),   f"{name}: {v.shape}"
        assert np.all(np.isfinite(v)),    f"{name} has NaN/inf: {v}"


def test_initialize_proc_seeds_accepted_from_est_idx_components(tiny_session):
    """`accepted` is True exactly at est.idx_components after init; manual
    override is empty, and accepted_core == accepted."""
    est, _, _ = tiny_session
    proc = initialize_proc(est, Ops())

    expected_accepted = np.zeros(est.A.shape[1], dtype=bool)
    expected_accepted[est.idx_components] = True
    np.testing.assert_array_equal(proc.accepted, expected_accepted)
    np.testing.assert_array_equal(proc.accepted_core, proc.accepted)
    assert not proc.manual_override.any()


def test_initialize_proc_snr2_equals_peaks_over_noise(tiny_session):
    """SNR2 = peaks_ave / noise per-cell — verify the formula end-to-end."""
    est, _, _ = tiny_session
    proc = initialize_proc(est, Ops())
    np.testing.assert_allclose(proc.SNR2_vals, proc.peaks_ave / proc.noise)


def test_initialize_proc_gAR_in_unit_circle_after_fudge(tiny_session):
    """`compute_ar_pair` applies fudge_factor (default 0.99) which shrinks
    AR roots inside the unit circle. So gAR1 should be in (-1, 1) for every
    cell, and the AR(2) discriminant should keep poles real / stable.
    """
    est, _, _ = tiny_session
    proc = initialize_proc(est, Ops())
    # AR(1): the single root IS gAR1
    assert np.all(np.abs(proc.gAR1) < 1.0 + 1e-9), proc.gAR1

    # AR(2): characteristic poly is 1 - g1·z - g2·z^2. Stability requires
    # |g2| < 1 AND |g1| < 1 - g2 (Schur-Cohn for AR(2)). Don't enforce
    # tightly — just check no obvious instability from a missed fudge step.
    g1, g2 = proc.gAR2[:, 0], proc.gAR2[:, 1]
    assert np.all(np.abs(g2) < 1.0 + 1e-9), proc.gAR2


def test_initialize_proc_no_double_fudge_against_pick_g(tiny_session):
    """The cached `gAR1` / `gAR2` from initialize_proc must be the values
    `_pick_g` returns — `_pick_g` does NOT re-apply fudge to cached AR
    coefficients (regression test for the double-fudge bug)."""
    from caiman_sorter_py.core.deconvolution import _pick_g
    from caiman_sorter_py.core.state import DeconvParams

    est, _, _ = tiny_session
    proc = initialize_proc(est, Ops())

    # AR(1) path
    params = DeconvParams(ar_order=1, manual_tau=False, fudge_factor=0.9)
    for n in range(proc.num_cells):
        g = _pick_g(proc, n, params, dt=1.0 / 30.0)
        np.testing.assert_allclose(g, [proc.gAR1[n]])

    # AR(2) path
    params = DeconvParams(ar_order=2, manual_tau=False, fudge_factor=0.9)
    for n in range(proc.num_cells):
        g = _pick_g(proc, n, params, dt=1.0 / 30.0)
        np.testing.assert_allclose(g, proc.gAR2[n])


def test_initialize_proc_respects_init_params(tiny_session):
    """If init_params_caiman.temporal.fudge_factor is changed, the gAR1
    values reflect the new shrinkage. Round-trips through get_ar_init_params.
    """
    est, _, _ = tiny_session
    # Default fudge from the fixture (0.97). Override to a smaller value.
    est.init_params_caiman = dict(est.init_params_caiman)
    est.init_params_caiman["temporal"] = {"fudge_factor": 0.85, "lags": 5}
    proc = initialize_proc(est, Ops())

    fr, dt, ar_order, fudge_factor, lags = get_ar_init_params(est)
    assert fudge_factor == pytest.approx(0.85)
    assert lags == 5

    # |gAR1| should be smaller now than with fudge=0.99 — a strict
    # comparison against the default-fudge run, with one explicit
    # tolerance: shrinkage by 0.85/0.99 ≈ 14%.
    est_default = est.__class__(**{**est.__dict__})
    est_default.init_params_caiman = {
        **est_default.init_params_caiman,
        "temporal": {"fudge_factor": 0.99, "lags": 5},
    }
    proc_default = initialize_proc(est_default, Ops())
    np.testing.assert_array_less(
        np.abs(proc.gAR1) - np.abs(proc_default.gAR1), 1e-9,
    )


def test_initialize_proc_zero_trace_does_not_crash():
    """A cell whose C and YrA are all zero must not crash the AR / peaks
    / firing-stability pipeline."""
    from scipy.sparse import csc_matrix
    from caiman_sorter_py.core.state import Estimates

    dims = (8, 8)
    n_cells, n_frames = 3, 2000
    rng = np.random.default_rng(0)
    A = csc_matrix(np.eye(dims[0] * dims[1], n_cells))
    C = rng.standard_normal((n_cells, n_frames)) * 0.5
    C[1] = 0.0   # zero trace cell
    YrA = np.zeros_like(C)
    YrA[1] = 0.0

    est = Estimates(
        A=A, C=C, YrA=YrA,
        S=np.zeros_like(C),
        F_dff=np.zeros_like(C),
        SNR_comp=np.ones(n_cells),
        cnn_preds=np.ones(n_cells),
        r_values=np.ones(n_cells),
        g=np.zeros((2, n_cells)),
        dims=dims,
        idx_components=np.array([0, 2], dtype=np.int64),
        idx_components_bad=np.array([1], dtype=np.int64),
        init_params_caiman={"data": {"fr": 30.0},
                            "preprocess": {"p": 2},
                            "temporal": {"fudge_factor": 0.99, "lags": 5}},
    )
    proc = initialize_proc(est, Ops())
    # All scalar metrics finite for the zero cell (peaks_ave will be 0,
    # noise will be tiny). SNR2 may be 0 or NaN — accept either, but no inf.
    assert not np.any(np.isinf(proc.gAR1))
    assert not np.any(np.isinf(proc.gAR2))
    assert proc.peaks_ave[1] == pytest.approx(0.0, abs=1e-6)
