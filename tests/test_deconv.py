"""Tests for core.deconvolution: fudge factor + smooth dF/dt.

The fudge-factor test is the load-bearing one: every other foopsi path
in the codebase passes g through apply_fudge_factor before handing it to
CaImAn. If apply_fudge_factor stops shrinking AR roots, the user-facing
'fudge_factor' GUI control silently does nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.deconvolution import (
    apply_fudge_factor,
    apply_smooth_dfdt_threshold,
    compute_smooth_dfdt,
    tau_to_g,
)
from caiman_sorter_py.core.deconvolution import _pick_g, _pick_gd_slow_decay
from caiman_sorter_py.core.state import DeconvParams, Proc
from caiman_sorter_py.core.state import SmoothDfdtParams


# ----------------------------------------------------------------------
# apply_fudge_factor
# ----------------------------------------------------------------------

def test_fudge_factor_one_is_identity():
    """fudge=1 (or numerically close) is a no-op."""
    g_in = np.array([0.95])
    out = apply_fudge_factor(g_in, 1.0)
    np.testing.assert_allclose(out, g_in)


def test_fudge_factor_ar1_shrinks_root():
    """For AR(1), g = [a], the fudged g is [fudge*a] exactly."""
    g_in = np.array([0.95])
    fudge = 0.9
    out = apply_fudge_factor(g_in, fudge)
    np.testing.assert_allclose(out, np.array([fudge * 0.95]))


@pytest.mark.parametrize("fudge", [0.99, 0.95, 0.9])
def test_fudge_factor_ar1_root_magnitude_shrinks_by_factor(fudge):
    """|root_after| / |root_before| ≈ fudge."""
    a = 0.92
    out = apply_fudge_factor(np.array([a]), fudge)
    assert float(out[0]) == pytest.approx(fudge * a)


def test_fudge_factor_ar2_real_roots():
    """For AR(2) with two real roots r1, r2: poly(fudge*r1, fudge*r2) gives
    g_new = [fudge*(r1+r2), -fudge^2 * r1 * r2] = [fudge*g1, fudge^2 * g2].
    """
    r1, r2 = 0.9, 0.6
    g_in = np.array([r1 + r2, -r1 * r2])
    fudge = 0.95
    out = apply_fudge_factor(g_in, fudge)
    expected = np.array([fudge * (r1 + r2), -(fudge ** 2) * r1 * r2])
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_fudge_factor_complex_roots_real_only():
    """When AR(2) has complex-conjugate roots, the function projects onto the
    real part before shrinking — verify the result stays real and the roots
    are shrunk in absolute value.
    """
    # AR(2) with complex roots: g1 = 0.5, g2 = -0.5
    g_in = np.array([0.5, -0.5])
    fudge = 0.9
    out = apply_fudge_factor(g_in, fudge)
    assert np.isreal(out).all()
    # Recover roots of the new poly: roots of (1 - g_new[0] z - g_new[1] z^2)
    roots = np.roots(np.concatenate([[1.0], -out]))
    # All root magnitudes should be <= the unshrunk magnitudes (fudge < 1)
    roots_in = np.roots(np.concatenate([[1.0], -g_in]))
    assert max(abs(roots)) < max(abs(roots_in))


# ----------------------------------------------------------------------
# tau_to_g
# ----------------------------------------------------------------------

def test_tau_to_g_ar1():
    g = tau_to_g(tau_decay=0.4, tau_rise=None, dt=1.0 / 30.0)
    assert g.shape == (1,)
    assert g[0] == pytest.approx(np.exp(-(1.0 / 30.0) / 0.4))


def test_tau_to_g_ar2_decay_only_when_rise_missing():
    """tau_rise=None should give AR(1) shape, not AR(2)."""
    g = tau_to_g(0.4, None, dt=1.0 / 30.0)
    assert g.shape == (1,)


def test_tau_to_g_ar2_matches_matlab_tau_c2d():
    """Matches MATLAB tau_c2d.m: poles at p1=1/τd and p2=1/τd + 1/τr.
    NOT the simpler 'two independent decay rates' form. The continuous
    impulse response interpreted by these g coefficients is the rise-decay
    envelope h(t) = (1 - exp(-t/τr)) * exp(-t/τd).
    """
    dt, tau_d, tau_r = 1.0 / 30.0, 0.4, 0.1
    g = tau_to_g(tau_decay=tau_d, tau_rise=tau_r, dt=dt)
    assert g.shape == (2,)
    p1, p2 = 1.0 / tau_d, 1.0 / tau_d + 1.0 / tau_r
    g_d = np.exp(-dt * p1); g_rd = np.exp(-dt * p2)
    np.testing.assert_allclose(g, [g_d + g_rd, -g_d * g_rd])


# ----------------------------------------------------------------------
# _pick_gd_slow_decay — mirrors MATLAB max(roots([1,-g]))
# ----------------------------------------------------------------------

def test_pick_gd_picks_signed_max_root():
    """For AR(2) with one positive and one negative real root, MATLAB picks
    the signed max (the positive one) — even if the negative root has a
    larger absolute value. The old Python code took max(abs(roots)) and
    diverged here."""
    # Construct g so the characteristic polynomial 1 - g0·z - g1·z^2 has
    # roots r1 = +0.3 and r2 = -0.8. Then g0 = r1+r2 = -0.5, g1 = -r1·r2 = 0.24.
    r1, r2 = 0.3, -0.8
    g = np.array([r1 + r2, -r1 * r2])
    gd = _pick_gd_slow_decay(g)
    assert gd == pytest.approx(r1)
    assert gd != pytest.approx(r2)


def test_pick_gd_two_positive_roots():
    r1, r2 = 0.9, 0.6
    g = np.array([r1 + r2, -r1 * r2])
    assert _pick_gd_slow_decay(g) == pytest.approx(max(r1, r2))


def test_pick_gd_ar1():
    g = np.array([0.95])
    assert _pick_gd_slow_decay(g) == pytest.approx(0.95)


# ----------------------------------------------------------------------
# _pick_g — fudge factor must NOT be re-applied to cached gAR1/gAR2
# (those are stored pre-fudged by initialize_proc; double-fudge diverges
# from MATLAB and silently shrinks AR roots twice).
# ----------------------------------------------------------------------

def test_pick_g_does_not_refudge_cached_ar1():
    proc = Proc(num_cells=1, num_frames=10)
    proc.gAR1 = np.array([0.95])   # already-fudged value cached in proc
    params = DeconvParams(ar_order=1, manual_tau=False, fudge_factor=0.9)
    g = _pick_g(proc, n_cell=0, params=params, dt=1.0 / 30.0)
    np.testing.assert_allclose(g, [0.95])   # NOT 0.95 * 0.9


def test_pick_g_does_not_refudge_cached_ar2():
    proc = Proc(num_cells=1, num_frames=10)
    proc.gAR2 = np.array([[1.5, -0.6]])
    params = DeconvParams(ar_order=2, manual_tau=False, fudge_factor=0.9)
    g = _pick_g(proc, n_cell=0, params=params, dt=1.0 / 30.0)
    np.testing.assert_allclose(g, [1.5, -0.6])


def test_pick_g_fudges_manual_tau():
    """Manual-tau path still applies fudge — there's no upstream cache there."""
    proc = Proc(num_cells=1, num_frames=10)
    params = DeconvParams(ar_order=1, manual_tau=True,
                          tau_decay=0.4, fudge_factor=0.9)
    dt = 1.0 / 30.0
    g = _pick_g(proc, n_cell=0, params=params, dt=dt)
    # Expected: AR(1) g = exp(-dt/τd), then fudged.
    raw = np.exp(-dt / 0.4)
    np.testing.assert_allclose(g, [0.9 * raw])


def test_pick_gd_complex_conjugate_pair():
    """Complex-conjugate roots collapse to their shared real part (matches
    CaImAn's apply_fudge_factor projection)."""
    # g0 = 1.0, g1 = -0.5 → roots z² - z + 0.5 → 0.5 ± 0.5i, real part 0.5
    g = np.array([1.0, -0.5])
    assert _pick_gd_slow_decay(g) == pytest.approx(0.5)


def test_tau_to_g_round_trip_with_ar_to_tau():
    """tau_to_g (matches tau_c2d) and ar_to_tau (matches tau_d2c) must
    invert each other under the MATLAB rise-envelope convention. This
    failed under the pre-fix Python convention where τr was treated as
    an independent decay rate."""
    from caiman_sorter_py.core.proc_init import ar_to_tau

    dt = 1.0 / 30.0
    for tau_d, tau_r in [(0.4, 0.1), (1.0, 0.05), (0.2, 0.02)]:
        g = tau_to_g(tau_decay=tau_d, tau_rise=tau_r, dt=dt)
        tau_rise_out, tau_decay_out = ar_to_tau(g, dt)
        assert tau_decay_out == pytest.approx(tau_d, rel=1e-6)
        assert tau_rise_out  == pytest.approx(tau_r, rel=1e-6)


# ----------------------------------------------------------------------
# compute_smooth_dfdt
# ----------------------------------------------------------------------

def test_smooth_dfdt_shape_preserved():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 500))
    out = compute_smooth_dfdt(data, fr=30.0, params=SmoothDfdtParams())
    assert out.shape == data.shape


def test_smooth_dfdt_zero_input_zero_output():
    data = np.zeros((4, 200))
    out = compute_smooth_dfdt(data, fr=30.0, params=SmoothDfdtParams())
    np.testing.assert_allclose(out, 0.0)


def test_smooth_dfdt_rectify_clips_negatives():
    rng = np.random.default_rng(1)
    data = np.cumsum(rng.standard_normal((3, 400)), axis=1)
    params = SmoothDfdtParams(rectify=True)
    out = compute_smooth_dfdt(data, fr=30.0, params=params)
    assert (out >= 0).all()


def test_smooth_dfdt_normalize_signed_max_matches_matlab():
    """MATLAB f_smooth_dfdt3 normalizes by signed max: `data / max(data)`.
    Peak (most positive value) becomes 1; sign of other values is preserved.
    """
    rng = np.random.default_rng(2)
    data = rng.standard_normal((3, 400))
    params = SmoothDfdtParams(normalize=True)
    out = compute_smooth_dfdt(data, fr=30.0, params=params)
    # Most positive value of each row is exactly 1 (MATLAB-matching).
    peaks = out.max(axis=1)
    np.testing.assert_allclose(peaks, 1.0)


def test_smooth_dfdt_order_normalize_then_rectify():
    """MATLAB order: smooth → normalize → rectify. After rectify clips
    negatives, the peak should still equal 1 (normalize happened first)."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((3, 400))
    params = SmoothDfdtParams(normalize=True, rectify=True)
    out = compute_smooth_dfdt(data, fr=30.0, params=params)
    assert (out >= 0).all()
    np.testing.assert_allclose(out.max(axis=1), 1.0)


# ----------------------------------------------------------------------
# apply_smooth_dfdt_threshold (Python-only post-process)
# ----------------------------------------------------------------------

def test_threshold_helper_is_shift_not_clip():
    """`apply_smooth_dfdt_threshold` implements `where(x > thr, x - thr, 0)` —
    survivors are shifted down by thr (no step discontinuity at thr)."""
    rng = np.random.default_rng(3)
    out_off = rng.standard_normal((5, 800))
    p = SmoothDfdtParams(apply_thresh=True, threshold_z=1.5)
    out_on = apply_smooth_dfdt_threshold(out_off, p)
    thr = 1.5 * out_off.std(axis=1, keepdims=True)
    expected = np.where(out_off > thr, out_off - thr, 0.0)
    np.testing.assert_allclose(out_on, expected)
    assert (out_on[out_off <= thr] == 0).all()


def test_threshold_helper_disabled_is_noop():
    rng = np.random.default_rng(4)
    out_in = rng.standard_normal((4, 400))
    p = SmoothDfdtParams(apply_thresh=False)
    out = apply_smooth_dfdt_threshold(out_in, p)
    np.testing.assert_array_equal(out, out_in)
