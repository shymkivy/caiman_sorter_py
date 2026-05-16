"""Deconvolution methods for calcium imaging traces.

Two methods supported:
  1. smooth_dfdt — vectorised Gaussian-smoothed temporal derivative.
                   Fast (~ms for 500 cells x 100k frames).
  2. foopsi      — Constrained OASIS via CaImAn, parallelised across cells
                   with joblib threads.

MCMC was supported in the MATLAB pipeline via the now-deprecated
`cont_ca_sampler` from CaImAn-MATLAB. Modern CaImAn Python ships no
equivalent; foopsi/OASIS is the recommended replacement.

Results are stored in `proc.smooth_dfdt` / `proc.foopsi` (DeconvResults)
as lists indexed by cell idx, matching what TracePanel reads.

Mirrors MATLAB: f_cs_deconvolution.m, f_cs_compute_smooth_dfof.m,
                f_cs_compute_constrained_foopsi.m (+ _core, _gather_params).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------
# Time-constant conversion
# ----------------------------------------------------------------------

def tau_to_g(tau_decay: float, tau_rise: Optional[float] = None,
             dt: float = 1.0) -> np.ndarray:
    """Convert continuous-time tau values to discrete AR coefficients.

    Mirrors MATLAB `tau_c2d.m`. The continuous impulse response is the
    rise-decay envelope
        h(t) = (1 - exp(-t / tau_rise)) * exp(-t / tau_decay)
    whose two poles in the s-plane are at
        p1 = 1 / tau_decay
        p2 = 1 / tau_decay + 1 / tau_rise
    (closed form of `eig([-(2/τd+1/τr), -(τr+τd)/(τr·τd²); 1 0])`).
    The discrete poles are `exp(-dt * p_i)`, so the AR(2) coefficients are
        g[0] = exp(-dt*p1) + exp(-dt*p2)
        g[1] = -exp(-dt*p1) * exp(-dt*p2)
    AR(1) (when `tau_rise` is missing): `g = [exp(-dt / tau_decay)]`.

    Args:
        tau_decay: Calcium decay tau in seconds.
        tau_rise:  Calcium rise tau in seconds. If None / <= 0 / inf, returns AR(1).
        dt:        Frame period in seconds.
    """
    p1  = 1.0 / tau_decay
    g_d = float(np.exp(-dt * p1))
    if tau_rise is None or tau_rise <= 0 or not np.isfinite(tau_rise):
        return np.array([g_d])
    p2   = p1 + 1.0 / tau_rise
    g_rd = float(np.exp(-dt * p2))
    return np.array([g_d + g_rd, -g_d * g_rd])


def apply_fudge_factor(g: np.ndarray, fudge: float) -> np.ndarray:
    """Shrink AR roots by `fudge` to reduce time-constant bias.

    Replicates CaImAn's `estimate_time_constant` recipe
    (caiman.source_extraction.cnmf.deconvolution lines 1007-1015):

        roots = np.roots([1, -g[0], -g[1], ...])
        roots = roots.real
        g_new = poly(fudge * roots), drop leading 1, negate

    For a stable AR(1) (`g = [a]`, root = a), this reduces to `g_new = [fudge*a]`.
    For AR(2) (`g = [g1, g2]`, roots r1, r2 with g1 = r1+r2, g2 = -r1*r2),
    `g_new = [fudge*g1, fudge^2 * g2]`.
    """
    g = np.asarray(g, dtype=float).flatten()
    if fudge >= 1.0 - 1e-12 or g.size == 0:
        return g
    roots = np.roots(np.concatenate([[1.0], -g]))
    roots = roots.real          # CaImAn does (r + r.conj) / 2  ==  r.real
    new_poly = np.poly(fudge * roots)
    return (-new_poly[1:]).real.astype(float)


def _pick_g(proc, n_cell: int, params, dt: float) -> Optional[np.ndarray]:
    """Choose AR coefficients for a cell.

    Order: manual tau → cached gAR1/gAR2 → None (let CaImAn estimate).

    `fudge_factor` is applied here ONLY for the manual-tau path. The cached
    `proc.gAR1` / `proc.gAR2` values were already fudged inside
    `initialize_proc` via `compute_ar_pair → estimate_ar_coefficients(...,
    fudge_factor=init_fudge)`. Re-applying fudge here would double-shrink
    the AR roots and diverge from the MATLAB pipeline (MATLAB's
    `estimate_time_constant.m` applies fudge once during init and the
    foopsi consumer reads pre-fudged g without re-fudging).
    """
    p = params.ar_order

    if params.manual_tau:
        g = (tau_to_g(params.tau_decay, None, dt) if p == 1
             else tau_to_g(params.tau_decay, params.tau_rise, dt))
        fudge = float(getattr(params, "fudge_factor", 1.0))
        return apply_fudge_factor(g, fudge)

    if p == 1 and proc.gAR1 is not None:
        return np.array([float(proc.gAR1[n_cell])])
    if p == 2 and proc.gAR2 is not None:
        return np.asarray(proc.gAR2[n_cell], dtype=float)

    return None     # caller will let CaImAn estimate from the trace


# ----------------------------------------------------------------------
# Helpers to ensure per-cell result lists are correctly sized
# ----------------------------------------------------------------------

def _ensure_lists(deconv_results, n_cells: int) -> None:
    """Resize DeconvResults.S/C/g to n_cells, preserving existing entries."""
    for attr in ("S", "C", "g"):
        lst = getattr(deconv_results, attr)
        if len(lst) != n_cells:
            new = [None] * n_cells
            for i, v in enumerate(lst[:n_cells]):
                new[i] = v
            setattr(deconv_results, attr, new)


# ----------------------------------------------------------------------
# 1. Smooth dF/dt  --  vectorised across cells
# ----------------------------------------------------------------------

def compute_smooth_dfdt(data: np.ndarray, fr: float, params) -> np.ndarray:
    """Smooth dF/dt computation — mirrors MATLAB f_smooth_dfdt3.m.

    Pipeline (matches `caiman_sorter/caiman_sorter_functions/deconvolution_dep/
    f_smooth_dfdt3.m`):
        deriv      = [0, diff(row)]
        smoothed   = gaussian_filter1d(deriv, sigma_frames)
        if normalize: smoothed / max(smoothed)   # signed max, matches MATLAB
        if rectify : max(smoothed, 0)

    The optional `apply_thresh` / `threshold_z` step is NOT applied here —
    it's a Python-only post-process available via `apply_smooth_dfdt_threshold`.

    Args:
        data:   (n_cells, n_frames) array of raw C + YrA traces.
        fr:     frame rate in Hz.
        params: SmoothDfdtParams (only gauss_sigma / normalize / rectify are
                read by this function).

    Returns:
        (n_cells, n_frames) smoothed-dF/dt array.
    """
    dt_ms = 1000.0 / fr
    sigma_frames = max(params.gauss_sigma / dt_ms, 0.5)

    deriv = np.diff(data, axis=1, prepend=data[:, :1])
    out   = gaussian_filter1d(deriv, sigma=sigma_frames, axis=1, mode="reflect")

    if params.normalize:
        # Signed max, matching MATLAB `temp_data / max(temp_data)`. Guard
        # against zero traces.
        peak = out.max(axis=1, keepdims=True)
        peak = np.where(peak == 0, 1.0, peak)
        out = out / peak

    if params.rectify:
        np.maximum(out, 0, out=out)

    return out


def apply_smooth_dfdt_threshold(out: np.ndarray, params) -> np.ndarray:
    """Optional post-process: zero values below `threshold_z * std` (per cell).

    No-op when `params.apply_thresh` is False. Survivors are shifted down by
    `thr` so the cut-off point sits at zero (no step discontinuity at `thr`).
    This step does not exist in MATLAB — it's a Python-only addition.
    """
    if not (params.apply_thresh and params.threshold_z > 0):
        return out
    std = out.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    thr = params.threshold_z * std
    return np.where(out > thr, out - thr, 0.0)


def run_smooth_dfdt(est, proc, ops,
                    cells: Optional[np.ndarray] = None,
                    log_cb: Optional[Callable[[str], None]] = None) -> None:
    """Compute smooth dF/dt for the given cells and store in proc.

    Output stored in proc.smooth_dfdt.S as a list of per-cell arrays.
    proc.smooth_dfdt_std is populated with sqrt(mean(positive^2)) per cell.
    """
    from caiman_sorter_py.core.state import get_init_param
    fr = float(get_init_param(est.init_params_caiman, "fr", 30))
    n_cells_total = est.A.shape[1]
    if cells is None:
        cells = np.arange(n_cells_total)
    cells = np.asarray(cells, dtype=int)

    data = est.C[cells] + est.YrA[cells]
    out  = compute_smooth_dfdt(data, fr, ops.smooth_dfdt)
    out  = apply_smooth_dfdt_threshold(out, ops.smooth_dfdt)

    _ensure_lists(proc.smooth_dfdt, n_cells_total)
    if proc.smooth_dfdt_std is None or len(proc.smooth_dfdt_std) != n_cells_total:
        proc.smooth_dfdt_std = np.zeros(n_cells_total, dtype=float)

    for i, n in enumerate(cells):
        n = int(n)
        proc.smooth_dfdt.S[n] = out[i].astype(np.float32, copy=False)
        pos = out[i][out[i] > 0]
        proc.smooth_dfdt_std[n] = float(np.sqrt(np.mean(pos**2))) if pos.size else 0.0

    if log_cb:
        dt_ms = 1000.0 / fr
        sigma_frames = max(ops.smooth_dfdt.gauss_sigma / dt_ms, 0.5)
        log_cb(f"smooth dF/dt: {len(cells)} cells, sigma = {sigma_frames:.1f} frames.")


# ----------------------------------------------------------------------
# 2. Constrained foopsi / OASIS
# ----------------------------------------------------------------------

_VALID_SOLVERS = ("oasis", "cvxpy", "cvx")

SOLVER_INSTALL_HINT = {
    "oasis": "",                                    # always available with caiman
    "cvxpy": "pip install cvxpy",
    "cvx":   "pip install cvxopt picos",
}


def solver_available(solver: str) -> bool:
    """Non-raising availability probe — returns True if the solver can be used."""
    try:
        _check_solver_available(solver)
        return True
    except Exception:
        return False


def _check_solver_available(solver: str) -> None:
    """Raise a friendly ImportError if the solver's dependencies are missing."""
    if solver == "oasis":
        return                       # ships with caiman
    if solver == "cvxpy":
        try:
            import cvxpy as _        # noqa: F401
        except ImportError:
            raise ImportError(
                "cvxpy solver requires the cvxpy package. "
                "Install with: pip install cvxpy"
            )
    elif solver == "cvx":
        try:
            import cvxopt as _       # noqa: F401
            import picos as _picos   # noqa: F401
        except ImportError:
            raise ImportError(
                "cvx solver requires cvxopt and picos. "
                "Install with: pip install cvxopt picos"
            )
    else:
        raise ValueError(
            f"Unknown solver {solver!r}. Choose one of: {_VALID_SOLVERS}"
        )


def _foopsi_one_cell(y: np.ndarray, g_init: Optional[np.ndarray],
                     sn: Optional[float], p: int,
                     solver: str = "oasis") -> tuple:
    """Run constrained foopsi for a single trace using the given solver.

    Available solvers (`method_deconvolution` in caiman):
      - 'oasis'  : Friedrich, Zhou & Paninski 2017 OASIS solver — fastest, O(T).
                   Recommended for everything.
      - 'cvxpy'  : Interior-point convex solver via cvxpy (ECOS/SCS). Slow,
                   O(T^3). Useful for verification or non-standard penalties.
      - 'cvx'    : Older cvxopt+picos implementation. Slowest, mostly historical.

    Note: CaImAn's `fudge_factor` argument is ignored when `g` is provided
    (it is only used inside `estimate_parameters`). We apply fudge_factor
    upstream in `_pick_g` so it always affects the AR coefficients that
    actually reach the solver.

    Reconstructs c = c + c1*gd^t + bl to match MATLAB f_cs_compute_constrained_foopsi_core.

    Returns:
        (c_full, sp, g_out) — denoised trace, deconvolved spikes, AR coeffs.
    """
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

    g_arg = None if g_init is None else np.asarray(g_init, dtype=float).flatten()

    c, bl, c1, g_out, sn_out, sp, _ = constrained_foopsi(
        y.astype(np.float64),
        g=g_arg, sn=sn, p=p,
        method_deconvolution=solver,
    )

    g_out_arr = np.asarray(g_out, dtype=float).flatten()
    gd = _pick_gd_slow_decay(g_out_arr)
    gd_vec = gd ** np.arange(len(y))
    c_full = np.asarray(c, dtype=float) + float(c1) * gd_vec + float(bl)
    return c_full, np.asarray(sp, dtype=float), g_out_arr


def _pick_gd_slow_decay(g: np.ndarray) -> float:
    """Return the dominant (slowest-decay) real root of `1 - g[0]z - g[1]z^2 - …`.

    Mirrors MATLAB `f_cs_compute_constrained_foopsi_core.m:5`:
        gd = max(roots([1, -g]))
    i.e. the SIGNED maximum (not max-of-abs). Roots are projected onto the
    real axis first — matching CaImAn's `estimate_time_constant` convention
    where complex-conjugate pairs collapse to their shared real part.
    """
    roots = np.roots(np.concatenate([[1.0], -np.asarray(g, dtype=float).flatten()]))
    return float(np.max(np.real(roots)))


def run_foopsi(est, proc, ops,
               cells: Optional[np.ndarray] = None,
               log_cb: Optional[Callable[[str], None]] = None,
               progress_cb: Optional[Callable[[int, int], None]] = None,
               parallel: bool = True) -> int:
    """Run constrained foopsi/OASIS for the given cells.

    Args:
        cells:       Cell indices to process (default: all).
        progress_cb: Called as (done, total) after each cell finishes.
        parallel:    If True, parallelise with joblib threads (falls back
                     to sequential if joblib is unavailable).

    Returns:
        Number of cells that succeeded.
    """
    from caiman_sorter_py.core.state import get_init_param
    fp = ops.foopsi
    fr = float(get_init_param(est.init_params_caiman, "fr", 30))
    dt = 1.0 / fr

    n_cells_total = est.A.shape[1]
    if cells is None:
        cells = np.arange(n_cells_total)
    cells = np.asarray(cells, dtype=int)

    solver = getattr(fp, "solver", "oasis")
    _check_solver_available(solver)            # fail fast if package missing
    if log_cb:
        log_cb(f"foopsi: solver = {solver}")

    _ensure_lists(proc.foopsi, n_cells_total)

    # Build per-cell job list
    jobs = []
    for n in cells:
        n   = int(n)
        y   = (est.C[n] + est.YrA[n]).astype(np.float64)
        sn  = float(proc.noise[n]) if proc.noise is not None else None
        g0  = _pick_g(proc, n, fp, dt)
        jobs.append((n, y, g0, sn, fp.ar_order))

    def _do(job):
        n, y, g0, sn, p = job
        try:
            c, sp, g_out = _foopsi_one_cell(y, g0, sn, p, solver=solver)
            return n, c, sp, g_out, None
        except Exception as exc:
            return n, None, None, None, repr(exc)

    use_parallel = parallel and len(jobs) > 4
    if use_parallel:
        try:
            from joblib import Parallel, delayed
            # `return_as="generator"` (joblib >= 1.3) yields per-completion so
            # progress_cb can fire incrementally instead of jumping from 0 to
            # 100% only after the whole batch finishes. Falls back to the
            # batched API if joblib is older.
            try:
                stream = Parallel(n_jobs=-1, prefer="threads",
                                  return_as="generator")(
                    delayed(_do)(job) for job in jobs
                )
                results = []
                for i, r in enumerate(stream):
                    results.append(r)
                    if progress_cb:
                        progress_cb(i + 1, len(jobs))
            except TypeError:
                # Older joblib without return_as — fall back to one-shot batch
                # and a single end-of-run progress call.
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_do)(job) for job in jobs
                )
                if progress_cb:
                    progress_cb(len(jobs), len(jobs))
        except ImportError:
            use_parallel = False

    if not use_parallel:
        results = []
        for i, job in enumerate(jobs):
            results.append(_do(job))
            if progress_cb:
                progress_cb(i + 1, len(jobs))

    n_ok = n_err = 0
    for n, c, sp, g_out, err in results:
        if err is not None:
            n_err += 1
            if log_cb and n_err <= 3:
                log_cb(f"foopsi: cell {n} failed ({err})")
            continue
        proc.foopsi.C[n] = c.astype(np.float32, copy=False)
        proc.foopsi.S[n] = sp.astype(np.float32, copy=False)
        proc.foopsi.g[n] = g_out
        n_ok += 1

    if log_cb:
        mode = "parallel" if use_parallel else "sequential"
        log_cb(f"foopsi: {n_ok}/{len(cells)} cells succeeded ({mode}).")
        if n_err > 3:
            log_cb(f"foopsi: {n_err - 3} more failures suppressed.")
    return n_ok
