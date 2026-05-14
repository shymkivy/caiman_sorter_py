"""Export a sort session as a MATLAB v7.3 .mat file.

The output mirrors `f_cs_save_data.m` from the MATLAB pipeline:
top-level variables `est`, `proc`, `ops` with the exact field names and shapes
that existing MATLAB analysis code expects.

Notes on conventions:
  - hdf5storage transposes shapes so MATLAB sees them in column-major form.
    Pass arrays in their natural MATLAB convention (e.g. (1, n_cells) row vector,
    (n_cells, n_frames) for time-series), not transposed.
  - Boolean → MATLAB logical, dtype=bool ndarray works.
  - Variable-length per-cell results → np.ndarray of dtype=object containing
    inner arrays. Empty cells become np.empty(0).
  - Sparse A is patched in manually after savemat (hdf5storage doesn't
    support scipy.sparse), using MATLAB_sparse attribute.
  - Contours and merged_cells are omitted (would require nested cell-of-struct).
    Downstream MATLAB code that needs contours can recompute from est.A.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import hdf5storage as hs
import numpy as np
from scipy.sparse import csc_matrix


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------

def save_session_mat(path: str | Path, est, proc, ops,
                     source_path: str = "",
                     fr: Optional[float] = None) -> None:
    """Write a MATLAB v7.3 .mat file mirroring f_cs_save_data.m output.

    Args:
        path:        Output .mat path.
        est, proc, ops: Dataclass instances from core.state.
        source_path: Source HDF5 path (informational, stored in /ops).
        fr:          Frame rate override; defaults to value from est.init_params_caiman.
    """
    path = Path(path)
    n_cells  = int(proc.num_cells)
    n_frames = int(proc.num_frames)

    if fr is None:
        from caiman_sorter_py.core.state import get_init_param
        fr = float(get_init_param(est.init_params_caiman, "fr", 30.0))

    data = {
        "est":  _build_est(est, n_cells, n_frames),
        "proc": _build_proc(proc, n_cells, n_frames),
        "ops":  _ops_with_init_params(_build_ops(ops, source_path), est),
    }

    # Write everything except the sparse A
    hs.savemat(str(path), data,
               format="7.3",
               matlab_compatible=True,
               store_python_metadata=False,
               compress=True,
               compression_algorithm="gzip")

    # Patch sparse A in manually
    _write_sparse_A(path, est.A)


# ----------------------------------------------------------------------
# /est
# ----------------------------------------------------------------------

def _build_est(est, n_cells: int, n_frames: int) -> dict:
    """Mirror MATLAB est struct (CaImAn estimates + a few sort fields).

    MATLAB shapes (matching reference _sort.mat dumped from the MATLAB pipeline):
      C/YrA/S/F_dff       : (n_cells, n_frames)
      SNR_comp/cnn_preds/r_values/neurons_sn : (n_cells, 1) column vectors
      g                   : (p, n_cells)  — row 1 is g1, row 2 is g2 for AR(2)
      dims                : (2, 1) column
      sn                  : (n_pixels, 1) column
    """
    out = {
        "C":     np.asarray(est.C,   dtype=np.float64),
        "YrA":   np.asarray(est.YrA, dtype=np.float64),
        "S":     np.asarray(est.S,   dtype=np.float64),
        "F_dff": np.asarray(est.F_dff, dtype=np.float64),
        "SNR_comp":  _col(est.SNR_comp, dtype=np.float64),
        "cnn_preds": _col(est.cnn_preds, dtype=np.float32),
        "r_values":  _col(est.r_values, dtype=np.float64),
        "dims":      _col(est.dims, dtype=np.int32),
        "idx_components":     _to_matlab_idx_col(est.idx_components),
        "idx_components_bad": _to_matlab_idx_col(est.idx_components_bad),
        "num_cells_original": float(est.num_cells_original or n_cells),
    }
    # AR coeffs g: our internal est.g is (p, n_cells); pass through unchanged
    # to match MATLAB convention.
    out["g"] = np.asarray(est.g, dtype=np.float64)

    # Optional fields (CaImAn HDF5 may omit some)
    if est.sn is not None:
        out["sn"] = _col(est.sn, dtype=np.float64)
    if est.b is not None:
        out["b"] = np.asarray(est.b, dtype=np.float64)
    if est.f is not None:
        out["f"] = np.asarray(est.f, dtype=np.float64)
    if est.neurons_sn is not None:
        out["neurons_sn"] = _col(est.neurons_sn, dtype=np.float64)
    if est.eval_params_caiman:
        out["eval_params_caiman"] = _rename_eval_keys(
            _flatten_dict_to_scalars(est.eval_params_caiman)
        )
    if est.init_params_caiman:
        out["init_params_caiman"] = _flatten_dict_to_scalars(est.init_params_caiman)

    # A is patched in separately after savemat — record a tiny placeholder so
    # the field exists in the struct's MATLAB_fields list.
    out["A"] = np.zeros((1, 1), dtype=np.float64)
    return out


# ----------------------------------------------------------------------
# /proc
# ----------------------------------------------------------------------

def _build_proc(proc, n_cells: int, n_frames: int) -> dict:
    """Mirror MATLAB proc struct.

    MATLAB shapes (column-major):
      comp_accepted, comp_accepted_core : (n_cells, 1) — column vector
      noise/peaks_ave/SNR2_vals/firing_stab_vals/skewness/num_zeros/gAR1/tauAR1
                       : (1, n_cells) row vectors
      gAR2, tauAR2     : (n_cells, 2)   — col 1 g/tau1, col 2 g/tau2
      idx_components   : (n_accepted, 1) int32 — 1-based MATLAB indices
      idx_manual       : (n_manual, 1)  double — 1-based MATLAB indices
    """
    accepted = np.asarray(proc.accepted, dtype=bool) if proc.accepted is not None \
               else np.zeros(n_cells, dtype=bool)
    manual = np.asarray(proc.manual_override, dtype=bool) if proc.manual_override is not None \
             else np.zeros(n_cells, dtype=bool)
    core   = np.asarray(proc.accepted_core, dtype=bool) if proc.accepted_core is not None \
             else accepted.copy()

    out = {
        "num_cells":  float(n_cells),
        "num_frames": float(n_frames),
        "dims":       _col(getattr(proc, "dims", (0, 0)) or (0, 0), dtype=np.int32),

        "comp_accepted":      accepted.reshape(-1, 1),               # column vector
        "comp_accepted_core": core.astype(np.float64).reshape(-1, 1),
        "idx_components":     _to_matlab_idx_col(np.where(accepted)[0]),
        "idx_components_bad": _to_matlab_idx_col(np.where(~accepted)[0]),
        # idx_manual & idx_manual_bad are double in MATLAB and need to be
        # explicitly 2D so hdf5storage transposes them correctly even when empty
        # (np 1-D arrays end up as 1-D HDF5 datasets, breaking MATLAB shape parity).
        "idx_manual":         np.atleast_2d(
            _to_matlab_idx_col(np.where(manual & accepted)[0]).astype(np.float64)
        ),
        "idx_manual_bad":     np.atleast_2d(
            _to_matlab_idx_col(np.where(manual & ~accepted)[0]).astype(np.float64)
        ),

        # Hard-coded peak detection params (matching MATLAB defaults)
        "peaks_to_ave":         5.0,
        "peak_bin_sig_size":    0.2,
        "peak_bin_zero_size":   5.0,
    }

    # Per-cell scalar metrics — column vectors (n_cells, 1) in MATLAB
    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "tauAR1"):
        v = getattr(proc, name, None)
        if v is not None:
            out[name] = _col(v, dtype=np.float64)

    # AR(2) coefficients — (n_cells, 2) in MATLAB (col 1 is g1, col 2 is g2)
    for name in ("gAR2", "tauAR2"):
        v = getattr(proc, name, None)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64)
            # If our internal shape is (2, n_cells), transpose to (n_cells, 2)
            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n_cells:
                arr = arr.T
            out[name] = arr

    out["deconv"] = _build_proc_deconv(proc, n_cells, n_frames)
    return out


def _build_proc_deconv(proc, n_cells: int, n_frames: int) -> dict:
    deconv: dict = {}

    # smooth_dfdt → dense (n_cells, n_frames) matrix + (n_cells, 1) std column vector
    sd_S = np.zeros((n_cells, n_frames), dtype=np.float64)
    for i in range(min(n_cells, len(proc.smooth_dfdt.S))):
        s = proc.smooth_dfdt.S[i]
        if s is not None:
            sd_S[i, :len(s)] = np.asarray(s, dtype=np.float64)

    # S_std is (1, n_cells) row vector in MATLAB — pass Python (1, n_cells) so
    # MATLAB sees a row vector (matching the reference _sort.mat shape).
    sd_std_src = (proc.smooth_dfdt_std if proc.smooth_dfdt_std is not None
                  else np.zeros(n_cells))
    sd_std = np.asarray(sd_std_src, dtype=np.float64).reshape(1, -1)

    deconv["smooth_dfdt"] = {
        "S":     sd_S,
        "S_std": sd_std,
    }

    # foopsi → cell arrays of per-cell results (empty for unprocessed)
    fS = np.empty(n_cells, dtype=object)
    fC = np.empty(n_cells, dtype=object)
    fg = np.empty(n_cells, dtype=object)
    fp = np.empty(n_cells, dtype=object)
    any_foopsi = False
    for i in range(n_cells):
        s = proc.foopsi.S[i] if i < len(proc.foopsi.S) else None
        c = proc.foopsi.C[i] if i < len(proc.foopsi.C) else None
        gi = proc.foopsi.g[i] if i < len(proc.foopsi.g) else None
        if s is None and c is None and gi is None:
            fS[i] = np.empty(0)
            fC[i] = np.empty(0)
            fg[i] = np.empty(0)
            fp[i] = np.empty(0)
            continue
        any_foopsi = True
        fS[i] = np.asarray(s, dtype=np.float64) if s is not None else np.empty(0)
        fC[i] = np.asarray(c, dtype=np.float64) if c is not None else np.empty(0)
        gi_arr = np.asarray(gi, dtype=np.float64) if gi is not None else np.empty(0)
        fg[i] = gi_arr
        fp[i] = float(gi_arr.size) if gi_arr.size else 1.0   # AR order per cell

    if any_foopsi:
        # MATLAB cell arrays for c_foopsi are (1, n_cells) row of cells
        deconv["c_foopsi"] = {
            "S": fS.reshape(1, -1),
            "C": fC.reshape(1, -1),
            "g": fg.reshape(1, -1),
            "p": fp.reshape(1, -1),
        }

    return deconv


# ----------------------------------------------------------------------
# /ops
# ----------------------------------------------------------------------

def _build_ops(ops, source_path: str) -> dict:
    """Mirror MATLAB ops struct produced by f_cs_collect_ops.m."""
    ec = ops.eval_caiman
    er = ops.eval_reject
    sd = ops.smooth_dfdt
    fp = ops.foopsi

    eval_params_caiman = {
        "SNR_thresh":         float(ec.snr_thresh),
        "SNR_lowest_thresh":  float(ec.snr_lowest_thresh),
        "cnn_thresh":         float(ec.cnn_thresh),
        "cnn_lowest_thresh":  float(ec.cnn_lowest_thresh),
        "rval_thresh":        float(ec.rval_thresh),
        "rval_lowest_thresh": float(ec.rval_lowest_thresh),
    }

    # MATLAB stores ops booleans as `double` (0.0 / 1.0), not `logical`.
    # We coerce all flag values to float so hdf5storage writes them as double
    # and the on-disk MATLAB_class matches the reference _sort.mat files.
    def _b(v) -> float:
        return 1.0 if bool(v) else 0.0

    eval_params2 = {
        # Thresholds
        "RejThrSNRCaiman":     float(er.snr_caiman),
        "RejThrSNR2":          float(er.snr2),
        "RejThrCNN":           float(er.cnn),
        "RejThrRvalues":       float(er.rvalues),
        "RejThrMinSigFrac":    float(er.min_sig_frac),
        "FiringStability":     float(er.firing_stability),
        "RejThrSkewness":      float(er.skewness),
        # Enable flags (stored as double 0/1 — matching the MATLAB pipeline)
        "EvalSNRcaiman":       _b(er.use_snr_caiman),
        "EvalSNR2":            _b(er.use_snr2),
        "EvalCNN":             _b(er.use_cnn),
        "EvalRvalues":         _b(er.use_rvalues),
        "EvalMinSigFrac":      _b(er.use_min_sig_frac),
        "EvalFiringStability": _b(er.use_firing_stability),
        "EvalSkewness":        _b(er.use_skewness),
    }

    deconv = {
        "smooth_dfdt": {
            "params": {
                "convolve_gaus":      1.0,
                "gauss_kernel_simga": float(sd.gauss_sigma),
                "rectify":            _b(sd.rectify),
                "normalize":          _b(sd.normalize),
                "apply_thresh":       _b(sd.apply_thresh),
                "threshold":          float(sd.threshold_z),
            },
            "gui": {
                "scale_value":    float(sd.scale),
                "shift_value":    float(sd.shift),
                "plot_threshold": _b(getattr(sd, "plot_threshold", False)),
            },
        },
        "c_foopsi": {
            "params": {
                "AR_val":             str(fp.ar_order),       # MATLAB stores '1' or '2'
                "manual_tau":         _b(fp.manual_tau),
                "manual_tau_rise":    float(fp.tau_rise),
                "manual_tau_decay":   float(fp.tau_decay),
                "convolve_gaus":      _b(fp.smooth_s),
                "gauss_kernel_simga": float(fp.smooth_sigma),
                "fudge_factor":       float(fp.fudge_factor),
                "solver":             str(fp.solver),
            },
            "gui": {
                "scale_value": float(fp.scale),
                "shift_value": float(fp.shift),
            },
        },
        # MCMC placeholder so legacy MATLAB code that reads ops.deconv.MCMC doesn't error
        "MCMC": {
            "params": {
                "AR_val":             "2",
                "B_param":            200.0,
                "Nsamples_param":     500.0,
                "manual_tau":         0.0,
                "manual_tau_rise":    0.1,
                "manual_tau_decay":   0.4,
                "convolve_gaus":      0.0,
                "gauss_kernel_simga": 50.0,
                "save_SAMP":          0.0,
            },
            "gui": {
                "scale_value": 1.0,
                "shift_value": 0.0,
            },
        },
    }

    # MATLAB SwitchCaimanEvaluate values: 'caiman evaluate' or 'reject threshold'
    switch_val = "caiman evaluate" if ops.eval_method == "caiman" else "reject threshold"

    out_ops = {
        "SwitchCaimanEvaluate": switch_val,
        "eval_params_caiman":   eval_params_caiman,
        "eval_params2":         eval_params2,
        "deconv":               deconv,
        "mat_file_loc":         str(source_path or ""),
    }
    return out_ops


def _ops_with_init_params(ops_dict: dict, est) -> dict:
    """Append a copy of est.init_params_caiman under ops, matching MATLAB layout.

    The MATLAB pipeline stores the full nested CaImAn parameter struct under
    `ops.init_params_caiman` (a copy of `est.init_params_caiman`). We mirror
    that so downstream MATLAB code that looks up e.g. `ops.init_params_caiman.data.fr`
    keeps working.
    """
    if est.init_params_caiman:
        ops_dict["init_params_caiman"] = _flatten_dict_to_scalars(est.init_params_caiman)
    return ops_dict


# ----------------------------------------------------------------------
# Sparse A
# ----------------------------------------------------------------------

def _write_sparse_A(path: Path, A: csc_matrix) -> None:
    """Replace /est/A with a MATLAB-style sparse group.

    MATLAB v7.3 sparse format:
        group attrs: MATLAB_class='double', MATLAB_sparse=<n_rows>
        datasets:
          data  (nnz,)     same dtype as A.data
          ir    (nnz,)     uint64 — row indices (= csc.indices)
          jc    (n_cols+1,) uint64 — column pointers (= csc.indptr)
    """
    A = A.tocsc()
    with h5py.File(path, "r+") as f:
        if "est/A" in f:
            del f["est/A"]
        grp = f["est"].create_group("A")
        grp.attrs["H5PATH"]        = b"/est"
        grp.attrs["MATLAB_class"]  = b"double"
        grp.attrs["MATLAB_sparse"] = np.uint64(A.shape[0])

        grp.create_dataset("data", data=A.data.astype(np.float64),
                           compression="gzip")
        grp.create_dataset("ir",   data=A.indices.astype(np.uint64),
                           compression="gzip")
        grp.create_dataset("jc",   data=A.indptr.astype(np.uint64),
                           compression="gzip")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _col(arr, dtype=np.float64) -> np.ndarray:
    """Force a 1-D array to (n, 1) column vector — MATLAB's convention for per-cell scalars."""
    if arr is None:
        return np.zeros((0, 1), dtype=dtype)
    return np.asarray(arr, dtype=dtype).reshape(-1, 1)


def _to_matlab_idx_col(idx) -> np.ndarray:
    """Convert a 0-based index array to (n, 1) int32 1-based MATLAB indices (column vector)."""
    if idx is None:
        return np.zeros((0, 1), dtype=np.int32)
    arr = np.asarray(idx, dtype=np.int64).flatten() + 1
    return arr.astype(np.int32).reshape(-1, 1)


_EVAL_KEY_RENAMES = {
    # CaImAn Python uses lowercase 'snr_*'; MATLAB pipeline uses 'SNR_*'.
    # cnn_* and rval_* already match.
    "snr_thresh":         "SNR_thresh",
    "snr_lowest_thresh":  "SNR_lowest_thresh",
}


def _rename_eval_keys(d: dict) -> dict:
    """Rename CaImAn-Python eval_params_caiman keys to match MATLAB output."""
    return {_EVAL_KEY_RENAMES.get(k, k): v for k, v in (d or {}).items()}


def _flatten_dict_to_scalars(d: dict) -> dict:
    """Recursively normalise a dict so values are scalars / arrays / nested dicts.

    Used for init_params_caiman / eval_params_caiman which may contain Python
    Nones, tuples, etc. We coerce types so hdf5storage can write them.
    """
    out: dict = {}
    for k, v in (d or {}).items():
        k = str(k)
        if v is None:
            out[k] = np.empty(0)
        elif isinstance(v, dict):
            out[k] = _flatten_dict_to_scalars(v)
        elif isinstance(v, (list, tuple)):
            try:
                out[k] = np.asarray(v)
            except Exception:
                out[k] = str(v)
        elif isinstance(v, bool):
            out[k] = bool(v)
        elif isinstance(v, (int, float, str)):
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = str(v)
    return out
