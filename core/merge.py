"""Duplicate-component detection and merging.

Mirrors MATLAB f_cs_find_similar_comp_core.m: pairs of components that
likely represent the same neuron are identified via
  (1) spatial overlap     — entries of A^T A above `spatial_thr`
  (2) temporal correlation — Pearson r of (C + YrA) above `temporal_thr`

The "choose best snr" method rejects the lower-SNR cell of each pair
(no new components are added — simple, reversible). The other MATLAB
methods (weighted ave, full mean corr, svd, nmf) construct a new merged
component; those are not yet implemented in the Python port.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHODS = ("choose best snr", "weighted ave", "full mean corr", "svd", "nmf")


@dataclass
class DuplicatePair:
    """A flagged duplicate-cell pair."""
    cell_a: int             # absolute index in est.A
    cell_b: int             # absolute index in est.A
    spatial_overlap: float  # A_a · A_b  (raw inner product, matches MATLAB AA)
    temporal_corr: float    # Pearson r of (C_a + YrA_a) and (C_b + YrA_b)


def find_duplicate_pairs(est, proc, ops) -> list[DuplicatePair]:
    """Detect duplicate components via spatial + temporal correlation.

    Args:
        est, proc, ops: standard dataclasses from core.state.

    Returns:
        Sorted list (descending spatial overlap) of pairs above both thresholds.
    """
    mp = ops.merge

    n_cells = est.A.shape[1]
    if mp.use_accepted_only and proc.accepted is not None:
        mask = np.asarray(proc.accepted, dtype=bool)
    else:
        mask = np.ones(n_cells, dtype=bool)

    idx_lut = np.where(mask)[0]
    if len(idx_lut) < 2:
        return []

    A_sub = est.A[:, idx_lut]                  # (n_pixels, n_acc)
    AA = (A_sub.T @ A_sub).toarray()           # (n_acc, n_acc) dense
    AA_lt = np.tril(AA, k=-1)                  # strict lower triangle

    rows, cols = np.where(AA_lt > mp.spatial_thr)
    if len(rows) == 0:
        return []

    trace = (est.C + est.YrA)[idx_lut]         # (n_acc, n_frames)
    t_centered = trace - trace.mean(axis=1, keepdims=True)
    t_norm = np.linalg.norm(t_centered, axis=1)
    t_norm[t_norm == 0] = 1.0

    pairs: list[DuplicatePair] = []
    for r, c in zip(rows, cols):
        corr = float((t_centered[r] @ t_centered[c]) / (t_norm[r] * t_norm[c]))
        if corr > mp.temporal_thr:
            pairs.append(DuplicatePair(
                cell_a=int(idx_lut[r]),
                cell_b=int(idx_lut[c]),
                spatial_overlap=float(AA_lt[r, c]),
                temporal_corr=corr,
            ))

    pairs.sort(key=lambda p: -p.spatial_overlap)
    return pairs


def sync_idx_components(est, proc) -> None:
    """Regenerate est.idx_components / idx_components_bad from proc.accepted.

    `proc.accepted` is the single source of truth for acceptance state.
    The cached index arrays in est must be rebuilt whenever it changes
    (after merge, manual accept/reject from bulk operations, etc.).
    """
    acc = np.asarray(proc.accepted, dtype=bool)
    est.idx_components     = np.where(acc)[0].astype(np.int64)
    est.idx_components_bad = np.where(~acc)[0].astype(np.int64)


def apply_choose_best_snr(proc, pairs: list[DuplicatePair]) -> dict:
    """Reject the lower-SNR2 cell of each pair, mark as manually overridden.

    Returns a dict:
        n_changed : number of cells whose `accepted` flag flipped to False
        kept      : list[int|None] — kept cell per input pair (None if SNR2 missing)
        rejected  : list[int|None] — rejected cell per input pair
    """
    if proc.SNR2_vals is None:
        raise ValueError("proc.SNR2_vals is unavailable — run full proc init first.")

    snr = np.asarray(proc.SNR2_vals)
    kept_list: list = []
    rej_list:  list = []
    n_changed = 0
    for p in pairs:
        if snr[p.cell_a] >= snr[p.cell_b]:
            kept, loser = p.cell_a, p.cell_b
        else:
            kept, loser = p.cell_b, p.cell_a
        kept_list.append(int(kept))
        rej_list.append(int(loser))
        if proc.accepted[loser]:
            proc.accepted[loser] = False
            proc.manual_override[loser] = True
            n_changed += 1
    return {"n_changed": n_changed, "kept": kept_list, "rejected": rej_list}


def weighted_ave_preview(est, pair: DuplicatePair) -> dict:
    """Build the visualisation payload for a merge-candidate pair.

    Combines the two cells using MATLAB's 'weighted ave' formula:
        A_merged = (A1 * ||tr1|| + A2 * ||tr2||) / (||tr1|| + ||tr2||)
        tr_merged = (tr1 * sum(A1) + tr2 * sum(A2)) / (sum(A1) + sum(A2))

    Returns a dict ready to pass into the plot:
        A1_2d, A2_2d, A_merged_2d : (h, w) np.ndarray
        crop                      : (r0, r1, c0, c1) bounding box (pixels)
        trace1, trace2, trace_merged : (n_frames,) np.ndarray
        fr                        : frame rate (Hz)
    """
    dims = est.dims
    a, b = pair.cell_a, pair.cell_b

    A1 = est.A[:, a].toarray().ravel()
    A2 = est.A[:, b].toarray().ravel()

    tr1 = (est.C[a] + est.YrA[a]).astype(float)
    tr2 = (est.C[b] + est.YrA[b]).astype(float)
    n1 = float(np.linalg.norm(tr1 - tr1.min()))
    n2 = float(np.linalg.norm(tr2 - tr2.min()))
    denom = n1 + n2 if (n1 + n2) > 0 else 1.0
    A_merged = (A1 * n1 + A2 * n2) / denom

    s1 = float(A1.sum())
    s2 = float(A2.sum())
    tdenom = s1 + s2 if (s1 + s2) > 0 else 1.0
    trace_merged = (tr1 * s1 + tr2 * s2) / tdenom

    A1_2d = A1.reshape(dims, order="F")
    A2_2d = A2.reshape(dims, order="F")
    Am_2d = A_merged.reshape(dims, order="F")

    # Bounding box around the union of non-zero footprints, with padding
    rows, cols = np.where((A1_2d + A2_2d) > 0)
    if rows.size:
        pad = 6
        r0, r1 = max(0, int(rows.min()) - pad), min(dims[0] - 1, int(rows.max()) + pad)
        c0, c1 = max(0, int(cols.min()) - pad), min(dims[1] - 1, int(cols.max()) + pad)
    else:
        r0, r1, c0, c1 = 0, dims[0] - 1, 0, dims[1] - 1

    from caiman_sorter_py.core.state import get_init_param
    fr = float(get_init_param(est.init_params_caiman, "fr", 30.0))
    return {
        "A1_2d": A1_2d, "A2_2d": A2_2d, "A_merged_2d": Am_2d,
        "crop": (r0, r1, c0, c1),
        "trace1": tr1, "trace2": tr2, "trace_merged": trace_merged,
        "fr": fr,
        "cell_a": a, "cell_b": b,
        "spatial_overlap": pair.spatial_overlap,
        "temporal_corr": pair.temporal_corr,
    }


# ----------------------------------------------------------------------
# Create-new-cell merge methods (mirrors f_cs_find_similar_comp_core.m)
# ----------------------------------------------------------------------

def _compute_merged_at(est, pair: DuplicatePair, method: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_new_flat, trace_new) for a duplicate pair using the given method.

    Mirrors the four MATLAB methods in f_cs_find_similar_comp_core.m.
    A_new_flat is shape (n_pixels,); trace_new is shape (n_frames,).
    """
    a, b = pair.cell_a, pair.cell_b
    A1 = est.A[:, a].toarray().ravel()
    A2 = est.A[:, b].toarray().ravel()
    tr1 = (est.C[a] + est.YrA[a]).astype(float)
    tr2 = (est.C[b] + est.YrA[b]).astype(float)

    if method == "weighted ave":
        n1 = float(np.linalg.norm(tr1 - tr1.min()))
        n2 = float(np.linalg.norm(tr2 - tr2.min()))
        td = n1 + n2 if (n1 + n2) > 0 else 1.0
        A_new = (A1 * n1 + A2 * n2) / td
        s1, s2 = float(A1.sum()), float(A2.sum())
        ad = s1 + s2 if (s1 + s2) > 0 else 1.0
        tr_new = (tr1 * s1 + tr2 * s2) / ad
        return A_new, tr_new

    # Methods below rebuild a (pixel, time) outer-product matrix on the
    # union of the two footprints, then pick the dominant component.
    A_mask = (A1 + A2) > 0
    full_comb = np.outer(A1[A_mask], tr1) + np.outer(A2[A_mask], tr2)

    if method == "full mean corr":
        full_trace = full_comb.sum(axis=0)
        nrm = np.linalg.norm(full_trace) or 1.0
        full_trace_n = full_trace / nrm
        A_compact = full_comb @ full_trace_n
        A_nrm = np.linalg.norm(A_compact) or 1.0
        A_new = np.zeros_like(A1)
        A_new[A_mask] = A_compact / A_nrm
        return A_new, full_trace_n * A_nrm

    if method == "svd":
        U, S_diag, Vt = np.linalg.svd(full_comb, full_matrices=False)
        A_new = np.zeros_like(A1)
        A_new[A_mask] = U[:, 0]
        return A_new, Vt[0] * S_diag[0]

    if method == "nmf":
        try:
            from sklearn.decomposition import NMF
        except ImportError as exc:
            raise ImportError(
                "NMF merge requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc
        full_pos = np.maximum(full_comb, 0)        # NMF needs non-negative input
        nmf = NMF(n_components=1, init="random", max_iter=200, tol=1e-4)
        W = nmf.fit_transform(full_pos)
        H = nmf.components_
        w_norm = np.linalg.norm(W[:, 0]) or 1.0
        A_new = np.zeros_like(A1)
        A_new[A_mask] = W[:, 0] / w_norm
        return A_new, H[0] * w_norm

    raise ValueError(f"Unknown merge method: {method!r}")


def _compute_contour_for(A_flat: np.ndarray, dims: tuple[int, int],
                          thr: float = 0.01) -> dict | None:
    """Compute a single contour dict for a new merged cell."""
    from scipy.sparse import csc_matrix
    from caiman_sorter_py.core.contours import compute_contours
    contours = compute_contours(
        csc_matrix(A_flat.reshape(-1, 1)), dims, thr=thr,
    )
    return contours[0] if contours else None


def _metrics_for_new_cell(trace: np.ndarray, S: np.ndarray, est, ops) -> dict:
    """Compute the full set of proc metrics for one freshly-created cell."""
    from scipy.stats import skew as scipy_skew
    from caiman_sorter_py.core.proc_init import (
        compute_noise, _batch_autocov, estimate_ar_coefficients,
        compute_peaks_ave, compute_firing_stability, ar_to_tau,
    )

    from caiman_sorter_py.core.state import get_init_param
    init = est.init_params_caiman
    fr           = float(get_init_param(init, "fr", 30.0))
    dt           = 1.0 / fr
    ar_order     = int(get_init_param(init, "ar_order", 2))
    fudge_factor = float(get_init_param(init, "fudge_factor", 0.99))
    lags         = int(get_init_param(init, "lags", 5))

    trace2d = trace[np.newaxis, :]
    S2d     = S[np.newaxis, :]

    noise_val = float(compute_noise(trace2d)[0])
    peaks_ave_val = float(compute_peaks_ave(trace2d, fr)[0])
    firing_stab_val = float(compute_firing_stability(S2d, fr)[0])
    skew_val = float(scipy_skew(trace))
    num_zeros_val = float(np.sum(trace == 0))
    snr2 = peaks_ave_val / noise_val if noise_val > 0 else 0.0

    total_lags = lags + max(ar_order, 1)
    acf = _batch_autocov(trace2d, total_lags)[0]
    g1 = estimate_ar_coefficients(1, noise_val, acf,
                                   lags=lags, fudge_factor=fudge_factor)
    g2 = estimate_ar_coefficients(ar_order, noise_val, acf,
                                   lags=lags, fudge_factor=fudge_factor)
    g2 = g2[:2] if len(g2) >= 2 else np.array([g2[0], 0.0])
    tau1 = ar_to_tau(g1, dt)
    tau2 = ar_to_tau(g2, dt)
    tau2 = tau2[:2] if len(tau2) >= 2 else np.array([0.0, tau2[0]])

    return {
        "noise":            noise_val,
        "skewness":         skew_val,
        "peaks_ave":        peaks_ave_val,
        "num_zeros":        num_zeros_val,
        "SNR2_vals":        snr2,
        "firing_stab_vals": firing_stab_val,
        "gAR1":             float(g1[0]),
        "gAR2":             np.asarray(g2, dtype=float),
        "tauAR1":           float(tau1[-1]),
        "tauAR2":           np.asarray(tau2, dtype=float),
    }


def _append_merged_cell(est, proc, *,
                        A_new: np.ndarray, c_new: np.ndarray, yra_new: np.ndarray,
                        sp_new: np.ndarray, g_new: np.ndarray,
                        metrics: dict, parent_a: int, parent_b: int,
                        foopsi_C: np.ndarray | None, foopsi_S: np.ndarray | None,
                        foopsi_g: np.ndarray | None,
                        contour: dict | None) -> int:
    """Append one new merged cell to all est/proc arrays. Returns its index."""
    from scipy.sparse import hstack, csc_matrix

    new_idx  = int(est.A.shape[1])
    n_frames = int(est.C.shape[1])

    # --- est arrays ---
    A_col = csc_matrix(A_new.reshape(-1, 1))
    est.A = hstack([est.A, A_col]).tocsc()
    est.C = np.vstack([est.C, c_new.reshape(1, -1).astype(est.C.dtype)])
    est.YrA = np.vstack([est.YrA, yra_new.reshape(1, -1).astype(est.YrA.dtype)])
    est.S = np.vstack([est.S, sp_new.reshape(1, -1).astype(est.S.dtype)])
    est.F_dff = np.vstack([
        est.F_dff,
        np.zeros((1, n_frames), dtype=est.F_dff.dtype),
    ])

    # Per-cell scalars: use mean/max of parents (matches MATLAB)
    snr_parents  = np.array([est.SNR_comp[parent_a], est.SNR_comp[parent_b]])
    cnn_parents  = np.array([est.cnn_preds[parent_a], est.cnn_preds[parent_b]])
    rval_parents = np.array([est.r_values[parent_a], est.r_values[parent_b]])
    est.SNR_comp  = np.append(est.SNR_comp,  float(np.nanmean(snr_parents)))
    est.cnn_preds = np.append(est.cnn_preds, float(np.nanmax(cnn_parents)))
    est.r_values  = np.append(est.r_values,  float(np.nanmax(rval_parents)))

    if est.neurons_sn is not None:
        est.neurons_sn = np.append(est.neurons_sn, float(metrics["noise"]))

    if est.g is not None and est.g.ndim == 2:
        p = est.g.shape[0]
        new_g = np.zeros((p, 1), dtype=est.g.dtype)
        flat  = np.asarray(g_new, dtype=float).flatten()
        new_g[:min(p, flat.size), 0] = flat[:p]
        est.g = np.hstack([est.g, new_g])

    if est.contours is not None:
        est.contours.append(contour or {"coordinates": None, "CoM": None})

    # Note: est.idx_components / idx_components_bad are regenerated from
    # proc.accepted at the end of apply_create_new (single source of truth).

    # --- proc arrays ---
    proc.num_cells       += 1
    proc.accepted         = np.append(proc.accepted,         True)
    proc.accepted_core    = np.append(proc.accepted_core,    False)
    proc.manual_override  = np.append(proc.manual_override,  True)
    proc.noise            = np.append(proc.noise,            metrics["noise"])
    proc.skewness         = np.append(proc.skewness,         metrics["skewness"])
    proc.peaks_ave        = np.append(proc.peaks_ave,        metrics["peaks_ave"])
    proc.num_zeros        = np.append(proc.num_zeros,        metrics["num_zeros"])
    proc.SNR2_vals        = np.append(proc.SNR2_vals,        metrics["SNR2_vals"])
    proc.firing_stab_vals = np.append(proc.firing_stab_vals, metrics["firing_stab_vals"])
    proc.gAR1             = np.append(proc.gAR1,             metrics["gAR1"])
    proc.tauAR1           = np.append(proc.tauAR1,           metrics["tauAR1"])
    proc.gAR2             = np.vstack([proc.gAR2,
                                       metrics["gAR2"].reshape(1, -1)])
    proc.tauAR2           = np.vstack([proc.tauAR2,
                                       metrics["tauAR2"].reshape(1, -1)])

    # DeconvResults lists — pad to current n_cells then write the new entries.
    # These lists are sparse: cells without deconv results have None at their index.
    def _pad(lst, length, fill=None):
        while len(lst) < length:
            lst.append(fill)

    target = int(proc.num_cells)
    for lst in (proc.smooth_dfdt.S, proc.smooth_dfdt.C, proc.smooth_dfdt.g,
                proc.foopsi.S, proc.foopsi.C, proc.foopsi.g):
        _pad(lst, target)

    if proc.smooth_dfdt_std is not None:
        if len(proc.smooth_dfdt_std) < target:
            proc.smooth_dfdt_std = np.concatenate([
                proc.smooth_dfdt_std,
                np.zeros(target - len(proc.smooth_dfdt_std)),
            ])

    proc.foopsi.S[new_idx] = foopsi_S
    proc.foopsi.C[new_idx] = foopsi_C
    proc.foopsi.g[new_idx] = foopsi_g

    return new_idx


def apply_create_new(est, proc, ops, pairs: list[DuplicatePair],
                     log_cb=None) -> dict:
    """Create a new merged cell for each pair, recompute all metrics, reject inputs.

    For each pair this:
      1. Builds A_new, trace_new via the chosen merge method.
      2. Runs constrained foopsi on trace_new (using the configured solver/AR order).
      3. Appends the new cell to est (A/C/YrA/S/F_dff/SNR_comp/cnn_preds/r_values/g/contours)
         and to proc (all metric arrays + DeconvResults entries).
      4. Marks the new cell `accepted=True`, `manual_override=True`.
      5. Marks both parents `accepted=False`, `manual_override=True`.

    The new cells survive future re-evaluation (`update_accepted` skips
    manual_override flags).
    """
    method = ops.merge.method
    if method == "choose best snr":
        return apply_choose_best_snr(proc, pairs)

    from caiman_sorter_py.core.deconvolution import _foopsi_one_cell

    kept_list: list = []
    rej_list:  list = []
    fp_p = int(getattr(ops.foopsi, "ar_order", 1))
    fp_solver = getattr(ops.foopsi, "solver", "oasis")
    contour_thr = float(getattr(ops, "contour_thr", 0.01))

    for i, pair in enumerate(pairs):
        if log_cb:
            log_cb(f"Merging pair {i + 1}/{len(pairs)}: cells {pair.cell_a} + {pair.cell_b}")

        A_new, trace_new = _compute_merged_at(est, pair, method)

        # Run foopsi to get C, S, g for the new cell (no g_init → caiman estimates)
        try:
            c_new, sp_new, g_new = _foopsi_one_cell(
                trace_new.astype(np.float64), g_init=None, sn=None,
                p=fp_p, solver=fp_solver,
            )
        except Exception as exc:
            if log_cb:
                log_cb(f"  foopsi failed ({exc!r}); falling back to raw trace.")
            c_new = trace_new.copy()
            sp_new = np.zeros_like(trace_new)
            g_new = np.array([0.95])

        yra_new = trace_new - c_new
        metrics = _metrics_for_new_cell(trace_new, sp_new, est, ops)
        contour = _compute_contour_for(A_new, est.dims, thr=contour_thr)

        new_idx = _append_merged_cell(
            est, proc,
            A_new=A_new, c_new=c_new, yra_new=yra_new,
            sp_new=sp_new, g_new=g_new,
            metrics=metrics,
            parent_a=pair.cell_a, parent_b=pair.cell_b,
            foopsi_C=c_new.astype(np.float32),
            foopsi_S=sp_new.astype(np.float32),
            foopsi_g=g_new,
            contour=contour,
        )

        # Reject the two parents (preserve existing rejections idempotently)
        proc.accepted[pair.cell_a] = False
        proc.accepted[pair.cell_b] = False
        proc.manual_override[pair.cell_a] = True
        proc.manual_override[pair.cell_b] = True

        kept_list.append(new_idx)
        rej_list.append((int(pair.cell_a), int(pair.cell_b)))

    return {"n_changed": len(kept_list), "kept": kept_list, "rejected": rej_list}


def apply_merge(est, proc, ops, pairs: list[DuplicatePair],
                log_cb=None) -> dict:
    """Dispatch the chosen merge method.

    'choose best snr' is reversible (only flips acceptance flags, no new cells).
    The other four methods create a new merged component per pair via
    apply_create_new, matching MATLAB f_cs_find_similar_comp_core.m.

    Also re-syncs est.idx_components / est.idx_components_bad from proc.accepted
    so downstream consumers (.mat export, session save) see consistent indices.
    """
    method = ops.merge.method
    if method == "choose best snr":
        result = apply_choose_best_snr(proc, pairs)
    else:
        result = apply_create_new(est, proc, ops, pairs, log_cb=log_cb)
    sync_idx_components(est, proc)
    return result
