"""Save and load curation sessions as HDF5.

Two file types, both HDF5:
  - Full session (*.h5):    est + proc + ops, self-contained — no source HDF5 needed.
  - Ops only    (*_ops.h5): only the /ops group, for sharing params across sessions.

File detection via root attribute /format ∈ { 'caiman_sorter_session', 'caiman_sorter_ops' }
— filenames don't have to end in any specific suffix; the format attribute is authoritative.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.sparse import csc_matrix

from caiman_sorter_py import __version__
from caiman_sorter_py.core.state import (
    DeconvResults, Estimates, OPS_SUB_PREFIXES, Ops, Proc,
)

FORMAT_SESSION = "caiman_sorter_session"
FORMAT_OPS     = "caiman_sorter_ops"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def save_session(path: str | Path, est: Estimates, proc: Proc, ops: Ops,
                 source_path: str = "") -> None:
    """Save est + proc + ops to a self-contained HDF5 session file."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["format"]      = FORMAT_SESSION
        f.attrs["app_version"] = __version__
        f.attrs["saved_at"]    = datetime.now().isoformat(timespec="seconds")
        f.attrs["source_path"] = _clean_str(str(source_path))

        _write_init_params(f.create_group("init_params_caiman"),
                           est.init_params_caiman or {})
        _write_init_params(f.create_group("eval_params_caiman"),
                           est.eval_params_caiman or {})
        _write_est(f.create_group("est"), est)
        _write_proc(f.create_group("proc"), proc)
        _write_ops(f.create_group("ops"), ops)


def load_session(path: str | Path) -> tuple[Estimates, Proc, Ops]:
    """Load a full session file (produced by save_session)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if fmt != FORMAT_SESSION:
            raise ValueError(
                f"{path.name}: format attribute is {fmt!r}, expected {FORMAT_SESSION!r}"
            )
        init_params = _read_init_params(f["init_params_caiman"]) if "init_params_caiman" in f else {}
        eval_params = _read_init_params(f["eval_params_caiman"]) if "eval_params_caiman" in f else {}
        est  = _read_est(f["est"], init_params, eval_params)
        proc = _read_proc(f["proc"], est.A.shape[1], est.C.shape[1])
        ops  = _read_ops(f["ops"])
    return est, proc, ops


def save_ops(path: str | Path, ops: Ops) -> None:
    """Save only the Ops dataclass to a small HDF5 file."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["format"]      = FORMAT_OPS
        f.attrs["app_version"] = __version__
        f.attrs["saved_at"]    = datetime.now().isoformat(timespec="seconds")
        _write_ops(f.create_group("ops"), ops)


def load_ops(path: str | Path) -> Ops:
    """Load an Ops-only file (produced by save_ops). Also accepts a full session file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "ops" not in f:
            raise ValueError(f"{path.name}: no /ops group found")
        return _read_ops(f["ops"])


def detect_format(path: str | Path) -> str:
    """Return one of: 'session', 'ops', 'caiman', 'session_mat', 'unknown'."""
    try:
        with h5py.File(path, "r") as f:
            fmt = f.attrs.get("format", "")
            if fmt == FORMAT_SESSION:
                return "session"
            if fmt == FORMAT_OPS:
                return "ops"
            # CaImAn HDF5 files have /estimates and /dims at top level
            if "estimates" in f and "dims" in f:
                return "caiman"
            # MATLAB-style sort .mat: /est, /proc, /ops at root
            if "est" in f and "proc" in f and "ops" in f:
                return "session_mat"
    except (OSError, KeyError):
        pass
    return "unknown"


# ----------------------------------------------------------------------
# Estimates  (we save only what TracePanel / ImagePanel / metrics need)
# ----------------------------------------------------------------------

def _write_est(g: h5py.Group, est: Estimates) -> None:
    """Write the parts of est needed to redisplay the session."""
    # Sparse A
    a = g.create_group("A")
    A = est.A.tocsc()
    a.create_dataset("data",    data=A.data,    compression="gzip")
    a.create_dataset("indices", data=A.indices, compression="gzip")
    a.create_dataset("indptr",  data=A.indptr,  compression="gzip")
    a.create_dataset("shape",   data=np.array(A.shape, dtype=np.int64))

    # Time-series (large) — gzip helps a lot for spikes (mostly zeros)
    _save_2d(g, "C",     est.C)
    _save_2d(g, "YrA",   est.YrA)
    _save_2d(g, "S",     est.S)
    _save_2d(g, "F_dff", est.F_dff)

    # Per-cell scalars
    g.create_dataset("SNR_comp",          data=np.asarray(est.SNR_comp))
    g.create_dataset("cnn_preds",         data=np.asarray(est.cnn_preds))
    g.create_dataset("r_values",          data=np.asarray(est.r_values))
    g.create_dataset("idx_components",    data=np.asarray(est.idx_components, dtype=np.int64))
    g.create_dataset("idx_components_bad",data=np.asarray(est.idx_components_bad, dtype=np.int64))
    if est.neurons_sn is not None:
        g.create_dataset("neurons_sn", data=np.asarray(est.neurons_sn))
    if est.sn is not None:
        g.create_dataset("sn", data=np.asarray(est.sn))
    # Spatial / temporal background — needed to reconstruct the "W comp + bkg"
    # image when re-displaying a saved session.
    if est.b is not None:
        g.create_dataset("b", data=np.asarray(est.b),
                         compression="gzip", chunks=True)
    if est.f is not None:
        g.create_dataset("f", data=np.asarray(est.f),
                         compression="gzip", chunks=True)

    # AR coefficient matrix from the original file
    g.create_dataset("g", data=np.asarray(est.g))

    g.create_dataset("dims", data=np.asarray(est.dims, dtype=np.int64))
    g.attrs["num_cells_original"] = int(est.num_cells_original or est.A.shape[1])


def _read_est(g: h5py.Group, init_params: dict, eval_params: dict) -> Estimates:
    A = csc_matrix(
        (g["A/data"][:], g["A/indices"][:], g["A/indptr"][:]),
        shape=tuple(int(x) for x in g["A/shape"][:]),
    )
    dims = tuple(int(x) for x in g["dims"][:])

    return Estimates.from_arrays(
        A=A, dims=dims,
        C=np.asarray(g["C"][:]),
        YrA=np.asarray(g["YrA"][:]),
        S=np.asarray(g["S"][:]),
        F_dff=np.asarray(g["F_dff"][:]),
        SNR_comp=np.asarray(g["SNR_comp"][:]),
        cnn_preds=np.asarray(g["cnn_preds"][:]),
        r_values=np.asarray(g["r_values"][:]),
        g=np.asarray(g["g"][:]),
        idx_components=np.asarray(g["idx_components"][:]),
        idx_components_bad=np.asarray(g["idx_components_bad"][:]),
        neurons_sn=np.asarray(g["neurons_sn"][:]) if "neurons_sn" in g else None,
        sn=np.asarray(g["sn"][:])                 if "sn"         in g else None,
        b=np.asarray(g["b"][:])                   if "b"          in g else None,
        f=np.asarray(g["f"][:])                   if "f"          in g else None,
        eval_params_caiman=eval_params,
        init_params_caiman=init_params,
        num_cells_original=int(g.attrs.get("num_cells_original", A.shape[1])),
    )


# ----------------------------------------------------------------------
# Proc
# ----------------------------------------------------------------------

def _write_proc(g: h5py.Group, proc: Proc) -> None:
    g.attrs["num_cells"]  = int(proc.num_cells)
    g.attrs["num_frames"] = int(proc.num_frames)

    for name in ("accepted", "accepted_core", "manual_override",
                 "noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "gAR2", "tauAR1", "tauAR2",
                 "smooth_dfdt_std"):
        v = getattr(proc, name, None)
        if v is not None:
            g.create_dataset(name, data=np.asarray(v))

    _write_deconv(g.create_group("smooth_dfdt"), proc.smooth_dfdt,
                  proc.num_cells, proc.num_frames)
    _write_deconv(g.create_group("foopsi"), proc.foopsi,
                  proc.num_cells, proc.num_frames)

    # Merge history (drives "Reset merges" undo across save/reload). Stored
    # as (n_merges, 2) int64 of (parent_a, parent_b). Skip the dataset when
    # the list is empty so old readers don't see an unexpected key.
    if proc.merge_parents:
        g.create_dataset(
            "merge_parents",
            data=np.asarray(proc.merge_parents, dtype=np.int64),
        )


def _read_proc(g: h5py.Group, n_cells: int, n_frames: int) -> Proc:
    proc = Proc(
        num_cells=int(g.attrs.get("num_cells", n_cells)),
        num_frames=int(g.attrs.get("num_frames", n_frames)),
    )
    for name in ("accepted", "accepted_core", "manual_override",
                 "noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "gAR2", "tauAR1", "tauAR2",
                 "smooth_dfdt_std"):
        if name in g:
            setattr(proc, name, np.asarray(g[name][:]))

    if "smooth_dfdt" in g:
        proc.smooth_dfdt = _read_deconv(g["smooth_dfdt"], n_cells, n_frames)
    if "foopsi" in g:
        proc.foopsi      = _read_deconv(g["foopsi"], n_cells, n_frames)

    # Merge history (absent on legacy saves — leaves the dataclass default
    # empty list in place).
    if "merge_parents" in g:
        arr = np.asarray(g["merge_parents"][:], dtype=np.int64)
        if arr.size:
            proc.merge_parents = [(int(a), int(b)) for a, b in arr.reshape(-1, 2)]
    return proc


def _write_deconv(g: h5py.Group, dr: DeconvResults,
                  n_cells: int, n_frames: int) -> None:
    """Store per-cell DeconvResults packed to only the populated cells.

    For a session with k of n_cells run, this writes:
      `idx`  (k,)            int32      cell indices that have data
      `S`    (k, n_frames)   float64    deconvolved spikes
      `C`    (k, n_frames)   float64    denoised calcium
      `g`    (k, max_p)      float64    AR coeffs, NaN-padded
      `done` (n_cells,)      bool       full mask (kept for back-compat readers)
    Packing avoids the (n_cells, n_frames) transient float64 buffer when only
    a handful of cells have been processed.
    """
    # Collect populated cell indices + sizes in one pass.
    idx_list: list[int] = []
    max_p = 0
    for i in range(n_cells):
        s = dr.S[i] if i < len(dr.S) else None
        c = dr.C[i] if i < len(dr.C) else None
        gi = dr.g[i] if i < len(dr.g) else None
        if s is None and c is None and gi is None:
            continue
        idx_list.append(i)
        if gi is not None:
            max_p = max(max_p, int(np.asarray(gi).size))

    k = len(idx_list)
    done = np.zeros(n_cells, dtype=bool)
    idx_arr = np.asarray(idx_list, dtype=np.int32)
    done[idx_arr] = True

    S_arr = np.zeros((k, n_frames), dtype=np.float64)
    C_arr = np.zeros((k, n_frames), dtype=np.float64)
    g_arr = np.full((k, max(max_p, 1)), np.nan, dtype=np.float64)

    for row, i in enumerate(idx_list):
        s = dr.S[i] if i < len(dr.S) else None
        c = dr.C[i] if i < len(dr.C) else None
        gi = dr.g[i] if i < len(dr.g) else None
        if s is not None:
            S_arr[row, :len(s)] = np.asarray(s, dtype=np.float64)
        if c is not None:
            C_arr[row, :len(c)] = np.asarray(c, dtype=np.float64)
        if gi is not None:
            gi_arr = np.asarray(gi, dtype=np.float64).flatten()
            g_arr[row, :gi_arr.size] = gi_arr

    g.create_dataset("idx",  data=idx_arr)
    g.create_dataset("S",    data=S_arr, compression="gzip", chunks=True)
    g.create_dataset("C",    data=C_arr, compression="gzip", chunks=True)
    g.create_dataset("g",    data=g_arr)
    g.create_dataset("done", data=done)


def _read_deconv(g: h5py.Group, n_cells: int, n_frames: int) -> DeconvResults:
    done = np.asarray(g["done"][:]) if "done" in g else np.zeros(n_cells, dtype=bool)
    S = g["S"][:] if "S" in g else None
    C = g["C"][:] if "C" in g else None
    G = g["g"][:] if "g" in g else None

    dr = DeconvResults(
        S=[None] * n_cells,
        C=[None] * n_cells,
        g=[None] * n_cells,
    )

    if "idx" in g:
        # Packed format: S/C/g are (k, n_frames), idx maps row → cell.
        idx_arr = np.asarray(g["idx"][:], dtype=np.int64)
        for row, i in enumerate(idx_arr):
            i = int(i)
            if i < 0 or i >= n_cells:
                continue
            if S is not None:
                dr.S[i] = S[row].copy()
            if C is not None:
                dr.C[i] = C[row].copy()
            if G is not None:
                gr = G[row]
                valid = gr[~np.isnan(gr)]
                dr.g[i] = valid if valid.size else None
        return dr

    # Legacy dense format: S/C/g are (n_cells, ...) indexed by cell id.
    for i in range(n_cells):
        if not done[i]:
            continue
        if S is not None:
            dr.S[i] = S[i].copy()
        if C is not None:
            dr.C[i] = C[i].copy()
        if G is not None:
            row = G[i]
            valid = row[~np.isnan(row)]
            dr.g[i] = valid if valid.size else None
    return dr


# ----------------------------------------------------------------------
# Ops
# ----------------------------------------------------------------------

def _write_ops(g: h5py.Group, ops: Ops) -> None:
    # Top-level scalar fields stored on the /ops group's attrs. (Not the full
    # OPS_TOP_FIELD_NAMES list — save_tag / save_as_mat are per-user prefs and
    # deliberately not embedded in the session file.)
    g.attrs["eval_method"]          = str(ops.eval_method)
    g.attrs["load_caiman_rejected"] = bool(ops.load_caiman_rejected)
    g.attrs["contour_thr"]          = float(ops.contour_thr)

    # Sub-dataclasses → nested groups. Iteration driven by the shared
    # OPS_SUB_PREFIXES registry in core/state.
    for name in OPS_SUB_PREFIXES:
        sub = getattr(ops, name)
        sg  = g.create_group(name)
        for fld in fields(sub):
            v = getattr(sub, fld.name)
            if isinstance(v, str):
                sg.attrs[fld.name] = v
            elif isinstance(v, bool):
                sg.attrs[fld.name] = bool(v)
            elif isinstance(v, (int, np.integer)):
                sg.attrs[fld.name] = int(v)
            else:
                sg.attrs[fld.name] = float(v)


def _read_ops(g: h5py.Group) -> Ops:
    ops = Ops()
    if "eval_method" in g.attrs:
        ops.eval_method = str(g.attrs["eval_method"])
    if "load_caiman_rejected" in g.attrs:
        ops.load_caiman_rejected = bool(g.attrs["load_caiman_rejected"])
    if "contour_thr" in g.attrs:
        ops.contour_thr = float(g.attrs["contour_thr"])

    for name in OPS_SUB_PREFIXES:
        if name not in g:
            continue
        sg = g[name]
        sub = getattr(ops, name)
        for fld in fields(sub):
            if fld.name not in sg.attrs:
                continue
            raw = sg.attrs[fld.name]
            if fld.type is bool or fld.type == "bool":
                setattr(sub, fld.name, bool(raw))
            elif fld.type is int or fld.type == "int":
                setattr(sub, fld.name, int(raw))
            elif fld.type is str or fld.type == "str":
                setattr(sub, fld.name, str(raw))
            else:
                setattr(sub, fld.name, float(raw))
    return ops


# ----------------------------------------------------------------------
# CaImAn init / eval params: store as nested attributes
# ----------------------------------------------------------------------

def _is_scalar(v) -> bool:
    return isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating))


def _clean_str(s) -> str:
    """Drop embedded NUL bytes from a string-like value.

    CaImAn HDF5 files store some params as fixed-length `|S32` byte strings,
    which leave trailing NULs after decode (`"2\\x00\\x00..."`). h5py's
    variable-length string type rejects embedded NULs on write — round-trip
    would crash with 'vlen strings do not support embedded nulls'.
    """
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode(errors="replace")
    if isinstance(s, str):
        return s.replace("\x00", "")
    return s


def _write_init_params(g: h5py.Group, params: dict) -> None:
    """Recursively write a nested dict of scalars/arrays to an HDF5 group."""
    for k, v in (params or {}).items():
        k = str(k)
        if isinstance(v, dict):
            _write_init_params(g.create_group(k), v)
        elif v is None:
            g.attrs[k] = "NoneType"
        elif _is_scalar(v):
            if isinstance(v, (bytes, np.bytes_, str)):
                v = _clean_str(v)
            try:
                g.attrs[k] = v
            except (TypeError, ValueError):
                g.attrs[k] = _clean_str(str(v))
        elif isinstance(v, np.ndarray):
            try:
                g.create_dataset(k, data=v)
            except (TypeError, ValueError):
                g.attrs[k] = _clean_str(str(v))
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype.kind in ("U", "S", "O"):
                    # Strip NULs from any string elements before vlen write.
                    cleaned = [_clean_str(x) for x in v]
                    g.create_dataset(k, data=np.asarray(cleaned, dtype=h5py.string_dtype()))
                else:
                    g.create_dataset(k, data=arr)
            except (TypeError, ValueError):
                g.attrs[k] = _clean_str(str(v))
        else:
            g.attrs[k] = _clean_str(str(v))


def _read_init_params(g: h5py.Group) -> dict:
    out: dict = {}
    for k, v in g.attrs.items():
        if isinstance(v, (bytes, np.bytes_)):
            s = _clean_str(v)
            out[str(k)] = None if s == "NoneType" else s
        elif isinstance(v, np.generic):
            out[str(k)] = v.item()
        elif isinstance(v, str):
            s = _clean_str(v)
            out[str(k)] = None if s == "NoneType" else s
        else:
            out[str(k)] = v
    for k in g.keys():
        item = g[k]
        if isinstance(item, h5py.Group):
            out[str(k)] = _read_init_params(item)
        else:
            out[str(k)] = np.asarray(item[:])
    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _save_2d(g: h5py.Group, name: str, arr: np.ndarray) -> None:
    """Save a 2-D array in its native dtype with gzip — preserves float64 from CaImAn."""
    arr = np.asarray(arr)
    g.create_dataset(name, data=arr, compression="gzip", chunks=True)
