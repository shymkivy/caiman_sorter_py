"""Tests for core.proc_init.compute_noise.

The estimator (Welch PSD in [0.25, 0.5] × Nyquist, logmexp) targets the
high-frequency noise floor — for AWGN of standard deviation sigma the
estimate should converge to sigma in expectation. We feed white noise
of known sigma and check that median(sn_hat) lies within tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.proc_init import compute_noise


@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 3.0])
def test_white_noise_estimate_matches_sigma(sigma):
    rng = np.random.default_rng(0)
    n_cells, n_frames = 50, 8000
    traces = rng.standard_normal((n_cells, n_frames)) * sigma
    sn = compute_noise(traces)
    assert sn.shape == (n_cells,)
    # logmexp estimator is unbiased for white noise; tolerance scales with N.
    median_sn = np.median(sn)
    assert median_sn == pytest.approx(sigma, rel=0.05), (
        f"sigma={sigma}: median sn = {median_sn} not within 5% "
        f"(min={sn.min():.4f}, max={sn.max():.4f})"
    )


def test_high_freq_signal_does_not_inflate_estimate():
    """Adding a low-frequency signal (in [0, 0.25] × Nyquist) should not
    leak into the noise estimate.
    """
    rng = np.random.default_rng(1)
    n_cells, n_frames = 20, 8000
    sigma = 0.2
    t = np.arange(n_frames)
    # Slow signal — well below 0.25 of the Nyquist band the estimator uses.
    signal = 2.0 * np.sin(2 * np.pi * 0.005 * t)
    traces = signal[None, :] + rng.standard_normal((n_cells, n_frames)) * sigma
    sn = compute_noise(traces)
    assert np.median(sn) == pytest.approx(sigma, rel=0.1)


def test_zero_trace_does_not_crash():
    """Edge case: a fully-zero trace shouldn't yield NaN/inf."""
    n_cells, n_frames = 5, 4000
    traces = np.zeros((n_cells, n_frames))
    sn = compute_noise(traces)
    assert np.all(np.isfinite(sn))
    # With PSD floored at 1e-15, sn floors at sqrt(exp(log(1e-15)/2)) ≈ 2.7e-8.
    assert np.all(sn < 1e-6)


def test_short_trace_still_returns_per_cell_array():
    """Even with very few frames the function must return shape (n_cells,)."""
    rng = np.random.default_rng(2)
    traces = rng.standard_normal((4, 64))
    sn = compute_noise(traces)
    assert sn.shape == (4,)
    assert np.all(np.isfinite(sn))
