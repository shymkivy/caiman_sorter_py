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

    # Vectorised Pearson correlation matrix on z-scored traces. The per-pair
    # Python loop is replaced with a single (n_acc, n_acc) matmul; for the
    # candidate spatial pairs we then index in once.
    trace = (est.C + est.YrA)[idx_lut]         # (n_acc, n_frames)
    t_centered = trace - trace.mean(axis=1, keepdims=True)
    t_norm = np.linalg.norm(t_centered, axis=1)
    t_norm[t_norm == 0] = 1.0
    t_z = t_centered / t_norm[:, np.newaxis]   # rows are unit-norm

    corr_pairs = np.einsum("ij,ij->i", t_z[rows], t_z[cols])  # (n_pair,)
    keep = corr_pairs > mp.temporal_thr
    if not keep.any():
        return []
    rows, cols, corr_pairs = rows[keep], cols[keep], corr_pairs[keep]
    overlaps = AA_lt[rows, cols]

    pairs = [
        DuplicatePair(
            cell_a=int(idx_lut[r]),
            cell_b=int(idx_lut[c]),
            spatial_overlap=float(o),
            temporal_corr=float(cc),
        )
        for r, c, o, cc in zip(rows, cols, overlaps, corr_pairs)
    ]
    pairs.sort(key=lambda p: -p.spatial_overlap)
    return pairs


def sync_idx_components(est, proc) -> None:
    """Overwrite est.idx_components / idx_components_bad with current state.

    `est.idx_components` is the pristine CaImAn-original record from load
    time and should generally not be mutated. This helper exists for ad-hoc
    use (one-off exports, tests) where a caller explicitly wants the
    current-state indices materialized into the est fields.

    Runtime sort flow does NOT call this; it derives current-state indices
    from `proc.accepted` directly via `np.where(proc.accepted)[0]`.
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
            # Also update accepted_core so the loser stays rejected even if
            # the user later clears manual overrides and re-evaluates: without
            # this, the stale accepted_core[loser]=True would bounce back into
            # accepted[loser] via evaluation.update_accepted.
            if proc.accepted_core is not None:
                proc.accepted_core[loser] = False
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
    # Baseline is min(C) only — matches MATLAB f_cs_find_similar_comp_core.m:28
    # (NOT min(C + YrA); the residual YrA is excluded from the baseline).
    base1 = float(est.C[a].min())
    base2 = float(est.C[b].min())
    n1 = float(np.linalg.norm(tr1 - base1))
    n2 = float(np.linalg.norm(tr2 - base2))
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
        # Baseline is min(C) only — matches MATLAB f_cs_find_similar_comp_core.m:28
        # (NOT min(C + YrA); the residual YrA is excluded from the baseline).
        base1 = float(est.C[a].min())
        base2 = float(est.C[b].min())
        n1 = float(np.linalg.norm(tr1 - base1))
        n2 = float(np.linalg.norm(tr2 - base2))
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


def _estimate_g_for_merged(trace: np.ndarray, est, p: int, fudge_factor: float
                           ) -> tuple[np.ndarray, float]:
    """Pre-estimate AR(p) coefficients + noise for a freshly-merged trace.

    Returns (g_fudged, sn). The fudge_factor is applied inside
    `estimate_ar_coefficients` (mirrors core.deconvolution._pick_g, which is
    the path used by every non-merge foopsi call). Feeding the result to
    `_foopsi_one_cell` as `g_init` ensures the user's fudge_factor is honored
    for merged cells — otherwise CaImAn re-estimates g internally and ignores
    the fudge factor entirely.
    """
    from caiman_sorter_py.core.proc_init import (
        _batch_autocov, compute_noise, estimate_ar_coefficients,
        get_ar_init_params,
    )
    _, _, _, _, lags = get_ar_init_params(est)

    trace2d = trace[np.newaxis, :]
    sn      = float(compute_noise(trace2d)[0])
    acf     = _batch_autocov(trace2d, lags + max(p, 1))[0]
    g       = estimate_ar_coefficients(p, sn, acf,
                                       lags=lags, fudge_factor=fudge_factor)
    return np.asarray(g, dtype=np.float64), sn


def _metrics_for_new_cell(trace: np.ndarray, S: np.ndarray, est, ops) -> dict:
    """Compute the full set of proc metrics for one freshly-created cell."""
    from scipy.stats import skew as scipy_skew
    from caiman_sorter_py.core.proc_init import (
        _batch_autocov, compute_ar_pair, compute_firing_stability, compute_noise,
        compute_peaks_ave, get_ar_init_params,
    )

    fr, dt, ar_order, fudge_factor, lags = get_ar_init_params(est)

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
    gAR1, gAR2, tauAR1, tauAR2 = compute_ar_pair(
        noise_val, acf, ar_order, lags, fudge_factor, dt,
    )

    return {
        "noise":            noise_val,
        "skewness":         skew_val,
        "peaks_ave":        peaks_ave_val,
        "num_zeros":        num_zeros_val,
        "SNR2_vals":        snr2,
        "firing_stab_vals": firing_stab_val,
        "gAR1":             gAR1,
        "gAR2":             gAR2,
        "tauAR1":           tauAR1,
        "tauAR2":           tauAR2,
    }


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

    fp_p = int(getattr(ops.foopsi, "ar_order", 1))
    fp_solver = getattr(ops.foopsi, "solver", "oasis")
    contour_thr = float(getattr(ops, "contour_thr", 0.01))

    # Collect per-pair records first; the actual array appends happen in one
    # batched pass at the end. Doing them inside the loop would mean N×hstack
    # on est.A and N×np.append on every per-cell array — O(n_cells × n_pairs).
    records: list[dict] = []
    rej_list: list = []
    for i, pair in enumerate(pairs):
        if log_cb:
            log_cb(f"Merging pair {i + 1}/{len(pairs)}: cells {pair.cell_a} + {pair.cell_b}")

        A_new, trace_new = _compute_merged_at(est, pair, method)

        # Pre-estimate AR coefficients for the merged trace and apply the user's
        # fudge_factor (mirrors core.deconvolution._pick_g). Without this the
        # foopsi call below passes g=None, CaImAn re-estimates g internally,
        # and the user's fudge_factor is silently ignored.
        g_init, sn_init = _estimate_g_for_merged(
            trace_new, est, fp_p, ops.foopsi.fudge_factor,
        )

        try:
            c_new, sp_new, g_new = _foopsi_one_cell(
                trace_new.astype(np.float64),
                g_init=g_init, sn=sn_init,
                p=fp_p, solver=fp_solver,
            )
        except Exception as exc:
            if log_cb:
                log_cb(f"  foopsi failed ({exc!r}); falling back to raw trace.")
            c_new = trace_new.copy()
            sp_new = np.zeros_like(trace_new)
            g_new = np.array([0.95])

        records.append(dict(
            pair=pair,
            A_new=A_new,
            c_new=c_new,
            yra_new=trace_new - c_new,
            sp_new=sp_new,
            g_new=g_new,
            metrics=_metrics_for_new_cell(trace_new, sp_new, est, ops),
            contour=_compute_contour_for(A_new, est.dims, thr=contour_thr),
        ))
        rej_list.append((int(pair.cell_a), int(pair.cell_b)))

    if not records:
        return {"n_changed": 0, "kept": [], "rejected": []}

    kept_list = _batch_append_merged_cells(est, proc, records)
    return {"n_changed": len(kept_list), "kept": kept_list, "rejected": rej_list}


def _batch_append_merged_cells(est, proc, records: list[dict]) -> list[int]:
    """Append N merged cells to est/proc with a single batched pass per array.

    Replaces N successive `_append_merged_cell` calls (which would reallocate
    every per-cell array N times — O(n_cells × n_pairs)). Returns the new
    indices in order.
    """
    from scipy.sparse import csc_matrix, hstack

    n_new = len(records)
    new_indices = list(range(int(est.A.shape[1]),
                             int(est.A.shape[1]) + n_new))
    n_frames = int(est.C.shape[1])

    # ---- est sparse / 2-D arrays --------------------------------------------
    new_cols = [csc_matrix(r["A_new"].reshape(-1, 1)) for r in records]
    est.A = hstack([est.A] + new_cols, format="csc")
    est.C     = np.vstack([est.C]   + [r["c_new"].reshape(1, -1).astype(est.C.dtype)   for r in records])
    est.YrA   = np.vstack([est.YrA] + [r["yra_new"].reshape(1, -1).astype(est.YrA.dtype) for r in records])
    est.S     = np.vstack([est.S]   + [r["sp_new"].reshape(1, -1).astype(est.S.dtype)  for r in records])
    est.F_dff = np.vstack([est.F_dff,
                            np.zeros((n_new, n_frames), dtype=est.F_dff.dtype)])

    # ---- est per-cell scalars (mean/max of parents — matches MATLAB) -------
    def _parent_scalar(arr, agg):
        vals = []
        for r in records:
            p = r["pair"]
            vals.append(float(agg(np.array([arr[p.cell_a], arr[p.cell_b]]))))
        return np.array(vals)

    est.SNR_comp  = np.concatenate([est.SNR_comp,  _parent_scalar(est.SNR_comp,  np.nanmean)])
    est.cnn_preds = np.concatenate([est.cnn_preds, _parent_scalar(est.cnn_preds, np.nanmax)])
    est.r_values  = np.concatenate([est.r_values,  _parent_scalar(est.r_values,  np.nanmax)])

    if est.neurons_sn is not None:
        est.neurons_sn = np.concatenate([
            est.neurons_sn,
            np.array([r["metrics"]["noise"] for r in records]),
        ])

    if est.g is not None and est.g.ndim == 2:
        p = est.g.shape[0]
        new_g = np.zeros((p, n_new), dtype=est.g.dtype)
        for j, r in enumerate(records):
            flat = np.asarray(r["g_new"], dtype=float).flatten()
            new_g[:min(p, flat.size), j] = flat[:p]
        est.g = np.hstack([est.g, new_g])

    if est.contours is not None:
        for r in records:
            est.contours.append(r["contour"] or {"coordinates": None, "CoM": None})

    # ---- proc 1-D arrays ---------------------------------------------------
    old_n = int(proc.num_cells)
    proc.num_cells += n_new
    proc.accepted        = np.concatenate([proc.accepted,        np.ones(n_new, dtype=bool)])
    proc.accepted_core   = np.concatenate([proc.accepted_core,   np.zeros(n_new, dtype=bool)])
    proc.manual_override = np.concatenate([proc.manual_override, np.ones(n_new, dtype=bool)])

    def _metric_col(key, dtype=np.float64):
        return np.array([r["metrics"][key] for r in records], dtype=dtype)

    def _extend_1d(attr_name: str, new_vals: np.ndarray) -> None:
        """Append new_vals to proc.<attr_name>; init to empty if None.

        Tolerates partially-initialised Proc (e.g. init_proc_minimal output
        where only `noise` is set). Without this, np.concatenate([None, x])
        raises and the merge appears to corrupt the state instead of failing
        cleanly.
        """
        cur = getattr(proc, attr_name)
        if cur is None:
            cur = np.full(old_n, np.nan, dtype=new_vals.dtype)
        setattr(proc, attr_name, np.concatenate([cur, new_vals]))

    _extend_1d("noise",            _metric_col("noise"))
    _extend_1d("skewness",         _metric_col("skewness"))
    _extend_1d("peaks_ave",        _metric_col("peaks_ave"))
    _extend_1d("num_zeros",        _metric_col("num_zeros"))
    _extend_1d("SNR2_vals",        _metric_col("SNR2_vals"))
    _extend_1d("firing_stab_vals", _metric_col("firing_stab_vals"))
    _extend_1d("gAR1",             _metric_col("gAR1"))
    _extend_1d("tauAR1",           _metric_col("tauAR1"))

    def _extend_2d(attr_name: str, new_rows: np.ndarray) -> None:
        cur = getattr(proc, attr_name)
        if cur is None:
            cur = np.full((old_n, new_rows.shape[1]), np.nan, dtype=new_rows.dtype)
        setattr(proc, attr_name, np.vstack([cur, new_rows]))

    _extend_2d("gAR2",   np.stack([r["metrics"]["gAR2"]   for r in records]))
    _extend_2d("tauAR2", np.stack([r["metrics"]["tauAR2"] for r in records]))

    # ---- DeconvResults lists ----------------------------------------------
    target = int(proc.num_cells)
    for lst in (proc.smooth_dfdt.S, proc.smooth_dfdt.C, proc.smooth_dfdt.g,
                proc.foopsi.S, proc.foopsi.C, proc.foopsi.g):
        while len(lst) < target:
            lst.append(None)
    if proc.smooth_dfdt_std is not None and len(proc.smooth_dfdt_std) < target:
        proc.smooth_dfdt_std = np.concatenate([
            proc.smooth_dfdt_std,
            np.zeros(target - len(proc.smooth_dfdt_std)),
        ])
    for new_idx, r in zip(new_indices, records):
        proc.foopsi.S[new_idx] = r["sp_new"].astype(np.float32)
        proc.foopsi.C[new_idx] = r["c_new"].astype(np.float32)
        proc.foopsi.g[new_idx] = r["g_new"]

    # ---- Parent acceptance flags ------------------------------------------
    parent_ids = np.array(
        [p for r in records for p in (r["pair"].cell_a, r["pair"].cell_b)]
    )
    proc.accepted[parent_ids]        = False
    proc.manual_override[parent_ids] = True

    # ---- Merge history (for "Reset merges" undo) --------------------------
    if proc.merge_parents is None:
        proc.merge_parents = []
    for r in records:
        proc.merge_parents.append(
            (int(r["pair"].cell_a), int(r["pair"].cell_b))
        )

    return new_indices


def reset_all_merges(est, proc) -> int:
    """Undo every merge that has been recorded on `proc.merge_parents`.

    Drops all merged-in cells (those at index >= est.num_cells_original) from
    every per-cell array on est and proc, then restores each merge parent's
    acceptance state to its auto-eval value (`proc.accepted_core[p]`) and
    clears its `manual_override` flag.

    No-op if `proc.merge_parents` is empty. Returns the number of merged
    cells that were undone.
    """
    if proc.merge_parents is None or len(proc.merge_parents) == 0:
        return 0

    n_orig = int(est.num_cells_original or proc.num_cells)
    n_undone = int(proc.num_cells) - n_orig
    if n_undone <= 0:
        proc.merge_parents = []
        return 0

    # ---- Truncate est ------------------------------------------------------
    est.A = est.A.tocsc()[:, :n_orig]
    est.C     = est.C[:n_orig]
    est.YrA   = est.YrA[:n_orig]
    est.S     = est.S[:n_orig]
    est.F_dff = est.F_dff[:n_orig]
    est.SNR_comp  = est.SNR_comp[:n_orig]
    est.cnn_preds = est.cnn_preds[:n_orig]
    est.r_values  = est.r_values[:n_orig]
    if est.neurons_sn is not None:
        est.neurons_sn = est.neurons_sn[:n_orig]
    if est.g is not None and est.g.ndim == 2:
        est.g = est.g[:, :n_orig]
    if est.contours is not None:
        del est.contours[n_orig:]

    # ---- Truncate proc per-cell arrays ------------------------------------
    proc.accepted        = proc.accepted[:n_orig]
    proc.accepted_core   = proc.accepted_core[:n_orig]
    proc.manual_override = proc.manual_override[:n_orig]
    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "tauAR1", "smooth_dfdt_std"):
        v = getattr(proc, name, None)
        if v is not None:
            setattr(proc, name, v[:n_orig])
    for name in ("gAR2", "tauAR2"):
        v = getattr(proc, name, None)
        if v is not None:
            setattr(proc, name, v[:n_orig])

    # ---- Truncate DeconvResults lists -------------------------------------
    for lst in (proc.smooth_dfdt.S, proc.smooth_dfdt.C, proc.smooth_dfdt.g,
                proc.foopsi.S,      proc.foopsi.C,      proc.foopsi.g):
        del lst[n_orig:]

    # ---- Restore parents to auto-eval state -------------------------------
    parents = {p for pair in proc.merge_parents for p in pair if p < n_orig}
    for p in parents:
        proc.accepted[p]        = bool(proc.accepted_core[p])
        proc.manual_override[p] = False

    # ---- Finalise ---------------------------------------------------------
    proc.num_cells = n_orig
    proc.merge_parents = []
    return n_undone


def apply_merge(est, proc, ops, pairs: list[DuplicatePair],
                log_cb=None) -> dict:
    """Dispatch the chosen merge method.

    'choose best snr' is reversible (only flips acceptance flags, no new cells).
    The other four methods create a new merged component per pair via
    apply_create_new, matching MATLAB f_cs_find_similar_comp_core.m.

    `est.idx_components` / `est.idx_components_bad` are NOT updated — they're
    the pristine CaImAn-original record from load time. Live acceptance state
    is `proc.accepted` (the single source of truth). Callers that need the
    current-state index arrays should derive them on demand via
    `np.where(proc.accepted)[0]`.
    """
    method = ops.merge.method
    if method == "choose best snr":
        result = apply_choose_best_snr(proc, pairs)
    else:
        result = apply_create_new(est, proc, ops, pairs, log_cb=log_cb)
    return result
