"""Automatic component evaluation logic.

Mirrors MATLAB: f_cs_evaluate_components.m
Two methods:
  1. 'caiman'           — union/intersection of SNR, CNN, r-value thresholds
  2. 'reject_threshold' — start with all accepted, apply per-metric cutoffs
"""
from __future__ import annotations

import numpy as np


def evaluate_components(est, proc, ops) -> np.ndarray:
    """Run automatic evaluation and return a boolean accepted mask.

    Does NOT mutate proc — caller assigns result to proc.accepted_core
    then calls update_accepted().
    """
    if ops.eval_method == "caiman":
        return _evaluate_caiman(est, ops.eval_caiman)
    return _evaluate_reject_threshold(est, proc, ops.eval_reject)


def _evaluate_caiman(est, params) -> np.ndarray:
    """CaImAn-style: union of cells passing any threshold, minus cells failing all lows."""
    n = est.C.shape[0]

    snr  = _safe(est.SNR_comp,  n)
    cnn  = _safe(est.cnn_preds, n)
    rval = _safe(est.r_values,  n)

    # Accept if above ANY of the three thresholds
    good = (snr  > params.snr_thresh) | \
           (cnn  >= params.cnn_thresh) | \
           (rval >= params.rval_thresh)

    # Reject if below ALL three lows (clearly bad)
    bad = (snr  <= params.snr_lowest_thresh) | \
          (cnn  <= params.cnn_lowest_thresh) | \
          (rval <= params.rval_lowest_thresh)

    return good & ~bad


def _evaluate_reject_threshold(est, proc, params) -> np.ndarray:
    """Reject-threshold: start all accepted, AND each enabled cutoff in sequence."""
    n = est.C.shape[0]
    accepted = np.ones(n, dtype=bool)

    if params.use_snr_caiman:
        accepted &= _safe(est.SNR_comp, n)  >= params.snr_caiman
    if params.use_snr2:
        accepted &= _safe(proc.SNR2_vals, n) >= params.snr2
    if params.use_cnn:
        accepted &= _safe(est.cnn_preds, n)  >= params.cnn
    if params.use_rvalues:
        accepted &= _safe(est.r_values, n)   >= params.rvalues
    if params.use_min_sig_frac and proc.num_zeros is not None:
        # Reject if fraction of zero frames exceeds (1 - min_sig_frac)
        accepted &= proc.num_zeros < (1.0 - params.min_sig_frac) * proc.num_frames
    if params.use_firing_stability:
        accepted &= _safe(proc.firing_stab_vals, n) >= params.firing_stability
    if params.use_skewness:
        accepted &= _safe(proc.skewness, n) >= params.skewness

    return accepted


def update_accepted(proc, core_mask: np.ndarray) -> None:
    """Apply core evaluation mask, preserving manually-overridden cells.

    Saves the current manually-set states, writes core_mask to proc.accepted,
    then restores only the cells flagged in proc.manual_override.
    """
    manual_states = proc.accepted[proc.manual_override].copy()

    proc.accepted_core = core_mask.copy()
    proc.accepted = core_mask.copy()

    proc.accepted[proc.manual_override] = manual_states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(arr, n: int) -> np.ndarray:
    """Return array as float64, or zeros if None/wrong length."""
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    return a if len(a) == n else np.zeros(n, dtype=np.float64)
