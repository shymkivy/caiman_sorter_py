"""Load a MATLAB v7.3 .mat session file produced by io/mat_export.save_session_mat
(or by the original MATLAB pipeline's f_cs_save_data.m).

.mat v7.3 is HDF5 under the hood. We read it directly with h5py rather than
scipy.io (which doesn't support v7.3) or hdf5storage (which doesn't transpose
cleanly for our shapes). The export format is documented in mat_export.py;
this loader inverts those conventions:

  - 2-D arrays appear in HDF5 as (cols, rows) of the MATLAB matrix → transpose
  - MATLAB column vectors are stored as (1, n) in HDF5 → flatten to (n,)
  - MATLAB char arrays are stored as uint16 with MATLAB_class='char' → decode
  - Sparse matrices: group with MATLAB_sparse=<n_rows> + data/ir/jc datasets
  - 1-based idx_components from MATLAB → 0-based for Python
  - ops flags stored as double 0/1 → cast back to bool
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from scipy.sparse import csc_matrix

from caiman_sorter_py.core.state import (
    DeconvResults, Estimates, MergeParams, Ops, Proc,
    SmoothDfdtParams, DeconvParams, EvalParamsCaiman, EvalParamsReject,
    SpikesParams,
)


def load_session_mat(path: str | Path) -> tuple[Estimates, Proc, Ops]:
    """Read a .mat v7.3 session file and return (est, proc, ops).

    Accepts any .mat that contains top-level /est, /proc, /ops groups in the
    layout written by mat_export.save_session_mat. Missing fields default to
    None / zeros / the dataclass default so a partial file still loads.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "est" not in f or "proc" not in f or "ops" not in f:
            raise ValueError(
                f"{path.name}: missing one of /est /proc /ops — not a sort .mat file"
            )
        est  = _read_est(f["est"])
        proc = _read_proc(f["proc"], est)
        ops  = _read_ops(f["ops"])
    return est, proc, ops


def detect_mat_session(path: str | Path) -> bool:
    """Return True if path is an HDF5 file with /est, /proc, /ops at root."""
    try:
        with h5py.File(path, "r") as f:
            return "est" in f and "proc" in f and "ops" in f
    except (OSError, KeyError):
        return False


# ======================================================================
# /est
# ======================================================================

def _read_est(g: h5py.Group) -> Estimates:
    A = _read_sparse(g["A"]) if "A" in g else None

    C     = _read_2d(g, "C")
    YrA   = _read_2d(g, "YrA")
    S     = _read_2d(g, "S")
    F_dff = _read_2d(g, "F_dff")

    SNR_comp  = _read_1d(g, "SNR_comp")
    cnn_preds = _read_1d(g, "cnn_preds", dtype=np.float32)
    r_values  = _read_1d(g, "r_values")
    sn        = _read_1d(g, "sn")
    # b and f are CaImAn background components. The legacy MATLAB pipeline
    # stores them transposed relative to CaImAn-Python's convention:
    #   on-disk (MATLAB internal): b: (n_bg, n_pixels)   f: (n_frames, n_bg)
    #   CaImAn-Python internal:    b: (n_pixels, n_bg)   f: (n_bg, n_frames)
    # io.mat_export writes them in MATLAB orientation; reverse the transpose
    # here so internal code (image panel, save_session) sees CaImAn shapes.
    bb        = _read_2d(g, "b")
    bg_f      = _read_2d(g, "f")
    if bb is not None and bb.ndim == 2:
        bb = bb.T
    if bg_f is not None and bg_f.ndim == 2:
        bg_f = bg_f.T
    neurons_sn = _read_1d(g, "neurons_sn")

    # AR coeffs: stored as (n_cells, p) in MATLAB, our internal is (p, n_cells)
    g_arr = _read_2d(g, "g")
    if g_arr is not None and g_arr.shape and g_arr.shape[0] != 0:
        # _read_2d already returns MATLAB-shape after transpose; ensure (p, n_cells)
        if g_arr.shape[1] == (SNR_comp.size if SNR_comp is not None else g_arr.shape[1]):
            pass  # already (p, n_cells)
        else:
            g_arr = g_arr.T

    dims_arr = _read_1d(g, "dims", dtype=np.int64)
    dims = tuple(int(x) for x in (dims_arr if dims_arr is not None else (0, 0)))[:2]

    idx_comp = _matlab_idx_to_python(_read_1d(g, "idx_components"))
    idx_bad  = _matlab_idx_to_python(_read_1d(g, "idx_components_bad"))

    eval_params = _read_struct(g["eval_params_caiman"]) if "eval_params_caiman" in g else {}
    init_params = _read_struct(g["init_params_caiman"]) if "init_params_caiman" in g else {}

    # Normalize eval_params back to lowercase keys used internally
    eval_params = _rename_eval_keys_inverse(eval_params)

    n_cells = (A.shape[1] if A is not None
               else (C.shape[0] if C is not None else 0))

    # If A is missing, fabricate an empty sparse matrix so the factory has a
    # consistent dims-aware backing for default-zero arrays.
    if A is None:
        A = csc_matrix((dims[0] * dims[1] if dims else 0, n_cells))

    return Estimates.from_arrays(
        A=A, dims=dims,
        C=C, YrA=YrA, S=S, F_dff=F_dff,
        SNR_comp=SNR_comp, cnn_preds=cnn_preds, r_values=r_values,
        g=g_arr,
        idx_components=idx_comp,
        idx_components_bad=idx_bad,
        sn=sn, b=bb, f=bg_f, neurons_sn=neurons_sn,
        eval_params_caiman=eval_params,
        init_params_caiman=init_params,
        num_cells_original=_read_scalar(g, "num_cells_original"),
    )


# ======================================================================
# /proc
# ======================================================================

def _read_proc(g: h5py.Group, est: Estimates) -> Proc:
    n_cells  = int(_read_scalar(g, "num_cells")  or est.C.shape[0])
    n_frames = int(_read_scalar(g, "num_frames") or est.C.shape[1])

    accepted = _read_bool_col(g, "comp_accepted",      n_cells)
    core     = _read_bool_col(g, "comp_accepted_core", n_cells)

    # idx_manual / idx_manual_bad are 1-based MATLAB indices.
    # manual_override is the union of those, in 0-based form.
    idx_manual     = _matlab_idx_to_python(_read_1d(g, "idx_manual"))
    idx_manual_bad = _matlab_idx_to_python(_read_1d(g, "idx_manual_bad"))
    manual = np.zeros(n_cells, dtype=bool)
    for arr in (idx_manual, idx_manual_bad):
        if arr is not None and arr.size:
            valid = (arr >= 0) & (arr < n_cells)
            manual[arr[valid]] = True

    proc = Proc(
        num_cells=n_cells,
        num_frames=n_frames,
        accepted=accepted,
        accepted_core=core,
        manual_override=manual,
    )

    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals", "gAR1", "tauAR1",
                 "smooth_dfdt_std"):
        v = _read_1d(g, name)
        if v is not None and v.size == n_cells:
            setattr(proc, name, v)

    # AR(2) coefficients stored (n_cells, 2) in MATLAB → keep that shape
    for name in ("gAR2", "tauAR2"):
        v = _read_2d(g, name)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64)
            if arr.shape == (2, n_cells):
                arr = arr.T
            setattr(proc, name, arr)

    if "deconv" in g:
        deconv = g["deconv"]
        if "smooth_dfdt" in deconv:
            sdg = deconv["smooth_dfdt"]
            S = _read_2d(sdg, "S")           # legacy slot = shaped (S_proc)
            S_raw = _read_2d(sdg, "S_raw")   # new; absent on pre-1.03 files
            std = _read_1d(sdg, "S_std")
            if S is not None and S.shape and S.size:
                # raw = S_raw when present, else fall back to the shaped trace.
                proc.smooth_dfdt = _dense_to_deconv_results(
                    S_raw if S_raw is not None and S_raw.size else S, S, n_cells)
            if std is not None and std.size == n_cells:
                proc.smooth_dfdt_std = std
        if "c_foopsi" in deconv:
            fp = deconv["c_foopsi"]
            proc.foopsi = _read_foopsi_cells(fp, n_cells)

    # Merge history (n_merges, 2) int32 1-based MATLAB indices → list of
    # 0-based (parent_a, parent_b) tuples. Absent on legacy .mat files.
    mp = _read_2d(g, "merge_parents")
    if mp is not None:
        arr = np.asarray(mp, dtype=np.int64)
        if arr.size and arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr - 1
            proc.merge_parents = [
                (int(a), int(b)) for a, b in arr if a >= 0 and b >= 0
            ]

    return proc


def _dense_to_deconv_results(S_raw: np.ndarray, S_proc: np.ndarray,
                             n_cells: int) -> DeconvResults:
    """Convert dense (n_cells, n_frames) raw + shaped matrices into per-cell lists.

    Used for smooth dF/dt (no C/g). Empty rows (all zeros) stay None so
    unprocessed cells are distinguishable. `S_proc` falls back to `S_raw`
    per-row when it is missing/empty.
    """
    dr = DeconvResults(S=[None] * n_cells, S_proc=[None] * n_cells,
                       C=[None] * n_cells, g=[None] * n_cells)
    for i in range(min(n_cells, S_raw.shape[0])):
        raw = S_raw[i]
        if not np.any(raw):
            continue
        dr.S[i] = raw.astype(np.float32)
        proc_row = S_proc[i] if S_proc is not None and i < S_proc.shape[0] else None
        dr.S_proc[i] = (proc_row if proc_row is not None and np.any(proc_row)
                        else raw).astype(np.float32)
    return dr


def _read_foopsi_cells(fp_group: h5py.Group, n_cells: int) -> DeconvResults:
    """Read c_foopsi MATLAB cell arrays (S, C, g, p) as per-cell lists.

    Each field in fp_group is a (1, n_cells) dataset of HDF5 object references
    pointing to the per-cell arrays.
    """
    dr = DeconvResults(S=[None] * n_cells, S_proc=[None] * n_cells,
                       C=[None] * n_cells, g=[None] * n_cells)
    f = fp_group.file

    def _cells_for(name: str) -> list:
        if name not in fp_group:
            return [None] * n_cells
        refs = fp_group[name][...]
        flat = np.asarray(refs).ravel()
        out: list = []
        for r in flat[:n_cells]:
            if not r:
                out.append(None); continue
            try:
                target = f[r]
                if _is_matlab_empty(target):
                    out.append(None); continue
                arr = np.asarray(target[...]).ravel()
                out.append(arr if arr.size else None)
            except Exception:
                out.append(None)
        while len(out) < n_cells:
            out.append(None)
        return out

    dr.S = _cells_for("S")
    dr.C = _cells_for("C")
    dr.g = _cells_for("g")
    # `S_proc` (new) = shaped spikes; pre-1.03 files lack it → fall back to raw.
    if "S_proc" in fp_group:
        dr.S_proc = _cells_for("S_proc")
    else:
        dr.S_proc = [s.copy() if s is not None else None for s in dr.S]
    return dr


# ======================================================================
# /ops
# ======================================================================

def _read_ops(g: h5py.Group) -> Ops:
    ops = Ops()

    switch = _read_string(g, "SwitchCaimanEvaluate")
    if switch:
        ops.eval_method = "caiman" if "caiman" in switch.lower() else "reject_threshold"

    if "eval_params_caiman" in g:
        ec = _read_struct(g["eval_params_caiman"])
        m = {
            "SNR_thresh":         "snr_thresh",
            "SNR_lowest_thresh":  "snr_lowest_thresh",
            "cnn_thresh":         "cnn_thresh",
            "cnn_lowest_thresh":  "cnn_lowest_thresh",
            "rval_thresh":        "rval_thresh",
            "rval_lowest_thresh": "rval_lowest_thresh",
        }
        for mat_k, field in m.items():
            if mat_k in ec:
                setattr(ops.eval_caiman, field, float(_scalar(ec[mat_k])))

    if "eval_params2" in g:
        er = _read_struct(g["eval_params2"])
        thr_map = {
            "RejThrSNRCaiman":  "snr_caiman",
            "RejThrSNR2":       "snr2",
            "RejThrCNN":        "cnn",
            "RejThrRvalues":    "rvalues",
            "RejThrMinSigFrac": "min_sig_frac",
            "FiringStability":  "firing_stability",
            "RejThrSkewness":   "skewness",
        }
        flag_map = {
            "EvalSNRcaiman":       "use_snr_caiman",
            "EvalSNR2":            "use_snr2",
            "EvalCNN":             "use_cnn",
            "EvalRvalues":         "use_rvalues",
            "EvalMinSigFrac":      "use_min_sig_frac",
            "EvalFiringStability": "use_firing_stability",
            "EvalSkewness":        "use_skewness",
        }
        for mat_k, field in thr_map.items():
            if mat_k in er:
                setattr(ops.eval_reject, field, float(_scalar(er[mat_k])))
        for mat_k, field in flag_map.items():
            if mat_k in er:
                setattr(ops.eval_reject, field, bool(_scalar(er[mat_k])))

    if "deconv" in g:
        dg = g["deconv"]
        if "smooth_dfdt" in dg:
            sd = ops.smooth_dfdt
            params = _read_struct(dg["smooth_dfdt/params"]) if "params" in dg["smooth_dfdt"] else {}
            gui    = _read_struct(dg["smooth_dfdt/gui"])    if "gui"    in dg["smooth_dfdt"] else {}
            if "gauss_kernel_simga" in params:
                sd.gauss_sigma = float(_scalar(params["gauss_kernel_simga"]))
            if "rectify"      in params: sd.rectify      = bool(_scalar(params["rectify"]))
            if "normalize"    in params: sd.normalize    = bool(_scalar(params["normalize"]))
            if "apply_thresh" in params: sd.apply_thresh = bool(_scalar(params["apply_thresh"]))
            if "threshold"    in params: sd.threshold_z  = float(_scalar(params["threshold"]))
            if "scale_value"  in gui:    sd.scale        = float(_scalar(gui["scale_value"]))
            if "shift_value"  in gui:    sd.shift        = float(_scalar(gui["shift_value"]))
            if "plot_threshold" in gui:  sd.plot_threshold = bool(_scalar(gui["plot_threshold"]))
        if "c_foopsi" in dg:
            fp = ops.foopsi
            params = _read_struct(dg["c_foopsi/params"]) if "params" in dg["c_foopsi"] else {}
            gui    = _read_struct(dg["c_foopsi/gui"])    if "gui"    in dg["c_foopsi"] else {}
            if "AR_val" in params:
                try:
                    fp.ar_order = int(_scalar(params["AR_val"]) or 1)
                except (TypeError, ValueError):
                    pass
            if "manual_tau"        in params: fp.manual_tau    = bool(_scalar(params["manual_tau"]))
            if "manual_tau_rise"   in params: fp.tau_rise      = float(_scalar(params["manual_tau_rise"]))
            if "manual_tau_decay"  in params: fp.tau_decay     = float(_scalar(params["manual_tau_decay"]))
            if "convolve_gaus"     in params: fp.smooth_s      = bool(_scalar(params["convolve_gaus"]))
            if "gauss_kernel_simga" in params: fp.smooth_sigma = float(_scalar(params["gauss_kernel_simga"]))
            if "fudge_factor"      in params: fp.fudge_factor  = float(_scalar(params["fudge_factor"]))
            if "solver"            in params:
                s = _scalar(params["solver"])
                if isinstance(s, str) and s:
                    fp.solver = s
            if "scale_value" in gui: fp.scale = float(_scalar(gui["scale_value"]))
            if "shift_value" in gui: fp.shift = float(_scalar(gui["shift_value"]))

    return ops


# ======================================================================
# Low-level readers
# ======================================================================

def _read_2d(g: h5py.Group, name: str) -> Optional[np.ndarray]:
    """Read a 2-D MATLAB dataset and return it in MATLAB orientation.

    HDF5 stores MATLAB matrices transposed, so we transpose on read.
    For (n_cells, n_frames) arrays MATLAB-side, the HDF5 dataset is
    (n_frames, n_cells); we return (n_cells, n_frames).
    """
    if name not in g:
        return None
    ds = g[name]
    if _is_matlab_empty(ds):
        return None
    arr = np.asarray(ds[...])
    if arr.ndim == 2:
        return arr.T
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _read_1d(g: h5py.Group, name: str,
             dtype: Optional[type] = None) -> Optional[np.ndarray]:
    """Read a 1-D MATLAB vector and return as a flat ndarray."""
    if name not in g:
        return None
    ds = g[name]
    if _is_matlab_empty(ds):
        return None
    arr = np.asarray(ds[...]).ravel()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _read_bool_col(g: h5py.Group, name: str, n_cells: int) -> Optional[np.ndarray]:
    v = _read_1d(g, name)
    if v is None:
        return None
    if v.size != n_cells:
        # MATLAB sometimes stores logical column vectors with leading singleton
        v = v.ravel()[:n_cells] if v.size >= n_cells else np.pad(v, (0, n_cells - v.size))
    return v.astype(bool)


def _read_scalar(g: h5py.Group, name: str):
    if name not in g:
        return None
    ds = g[name]
    if _is_matlab_empty(ds):
        return None
    v = np.asarray(ds[...]).ravel()
    return v.item() if v.size else None


def _read_string(g: h5py.Group, name: str) -> Optional[str]:
    if name not in g:
        return None
    ds = g[name]
    return _decode_matlab_char(ds)


def _read_sparse(group: h5py.Group) -> csc_matrix:
    """Decode a MATLAB v7.3 sparse group into a scipy CSC matrix."""
    n_rows = int(group.attrs.get("MATLAB_sparse", 0))
    data = np.asarray(group["data"][:])
    indices = np.asarray(group["ir"][:])
    indptr = np.asarray(group["jc"][:])
    n_cols = max(indptr.size - 1, 0)
    return csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def _matlab_idx_to_python(arr: Optional[np.ndarray]) -> np.ndarray:
    """1-based MATLAB index vector → 0-based int32 (empty if missing)."""
    if arr is None or arr.size == 0:
        return np.array([], dtype=np.int32)
    return (arr.astype(np.int64) - 1).astype(np.int32)


def _read_struct(g: h5py.Group) -> dict:
    """Recursively read a MATLAB struct group as a nested dict."""
    out: dict = {}
    for k in g.keys():
        item = g[k]
        if isinstance(item, h5py.Group):
            if "MATLAB_sparse" in item.attrs:
                # Sparse inside a struct — rare for us, keep as csc
                try:
                    out[str(k)] = _read_sparse(item)
                except Exception:
                    out[str(k)] = None
            else:
                out[str(k)] = _read_struct(item)
        else:
            out[str(k)] = _read_value(item)
    return out


def _read_value(ds: h5py.Dataset):
    """Decode a single HDF5 dataset to a Python scalar / ndarray / str."""
    if _is_matlab_empty(ds):
        return None
    cls = ds.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, (bytes, np.bytes_)) else str(cls)
    if cls == "char":
        return _decode_matlab_char(ds)

    arr = np.asarray(ds[...])
    if arr.size == 1:
        return arr.ravel()[0].item() if hasattr(arr.ravel()[0], "item") else arr.ravel()[0]
    if arr.ndim == 2:
        return arr.T
    return arr


def _decode_matlab_char(ds: h5py.Dataset) -> str:
    """MATLAB char arrays are uint16 column-major. Decode to a Python str."""
    if _is_matlab_empty(ds):
        return ""
    arr = np.asarray(ds[...])
    if arr.dtype.kind in ("S", "O"):
        # Already a byte string somehow
        raw = arr.ravel()
        return b"".join(b if isinstance(b, bytes) else b.encode()
                        for b in raw).decode(errors="replace")
    # uint16 char codes — MATLAB stores as (n_chars, 1) so flatten in F order
    return "".join(chr(int(c)) for c in arr.flatten(order="F"))


def _is_matlab_empty(ds) -> bool:
    """MATLAB marks empty arrays with MATLAB_empty=1 (and stores a dim array)."""
    try:
        if int(ds.attrs.get("MATLAB_empty", 0)) == 1:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _scalar(v: Any):
    """Coerce a possibly-array value down to a Python scalar."""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    arr = np.asarray(v).ravel()
    if arr.size == 0:
        return None
    item = arr[0]
    return item.item() if hasattr(item, "item") else item


def _rename_eval_keys_inverse(d: dict) -> dict:
    """Inverse of mat_export._rename_eval_keys (SNR_* → snr_*)."""
    rev = {
        "SNR_thresh":         "snr_thresh",
        "SNR_lowest_thresh":  "snr_lowest_thresh",
    }
    return {rev.get(k, k): v for k, v in (d or {}).items()}

