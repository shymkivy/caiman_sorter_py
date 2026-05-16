"""Load CaImAn HDF5 output files into an Estimates dataclass.

Mirrors MATLAB: f_cs_load_h5_est.m + f_cs_extract_h5_data.m
Does NOT use CNMF.load() — hand-rolled with h5py for version tolerance
and fast startup (no caiman import at load time).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csc_matrix, hstack

from caiman_sorter_py.core.state import Estimates


def load_hdf5(path: str | Path, load_rejected: bool = True,
              contour_thr: float = 0.01) -> Estimates:
    """Load a CaImAn HDF5 file and return an Estimates object.

    Handles both the main /estimates group and the optional
    /estimates/discarded_components group, combining them into a
    single Estimates with correct idx_components / idx_components_bad.

    Args:
        path: Path to the .hdf5 file produced by CaImAn.
        load_rejected: If False, skip the discarded_components group entirely,
            returning only the cells CaImAn originally accepted.

    Returns:
        Estimates dataclass populated from the file.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        dims = tuple(int(x) for x in f["dims"][:])  # (height, width)

        init_params = _load_init_params(f)
        eval_params = _load_eval_params(f)

        # --- main estimates group ---
        (A_m, C_m, YrA_m, S_m, F_dff_m,
         SNR_m, cnn_m, rval_m, g_m, nsn_m,
         idx_comp, idx_bad) = _load_estimates_group(f, "estimates")
        n_main = C_m.shape[0]

        # background from main group only (shared across all cells)
        sn   = np.array(f["estimates/sn"][:])   if _has_data(f, "estimates/sn")   else None
        b    = np.array(f["estimates/b"][:])     if _has_data(f, "estimates/b")    else None
        bg_f = np.array(f["estimates/f"][:])     if _has_data(f, "estimates/f")    else None

        # --- discarded components ---
        disc_path = "estimates/discarded_components"
        if load_rejected and disc_path in f and _has_data(f, f"{disc_path}/C"):
            (A_d, C_d, YrA_d, S_d, F_dff_d,
             SNR_d, cnn_d, rval_d, g_d, nsn_d,
             _, _) = _load_estimates_group(f, disc_path)
            n_disc = C_d.shape[0]
        else:
            n_disc = 0

    # --- combine main + discarded ---
    if n_disc > 0:
        A       = hstack([A_m, A_d]).tocsc()
        C       = np.vstack([C_m, C_d])
        YrA     = np.vstack([YrA_m, YrA_d])
        S       = np.vstack([S_m, S_d])
        F_dff   = np.vstack([F_dff_m, F_dff_d])
        SNR     = np.concatenate([SNR_m, SNR_d])
        cnn     = np.concatenate([cnn_m, cnn_d])
        rval    = np.concatenate([rval_m, rval_d])
        nsn     = np.concatenate([nsn_m, nsn_d])

        ar_order = g_m.shape[1]
        g_d_pad  = np.zeros((n_disc, ar_order)) if g_d.shape[1] == 0 else g_d
        g        = np.vstack([g_m, g_d_pad]).T          # → (ar_order, n_cells)

        idx_comp_bad = np.concatenate([
            idx_bad,
            n_main + np.arange(n_disc, dtype=np.int32),
        ])
    else:
        A    = A_m
        C    = C_m
        YrA  = YrA_m
        S    = S_m
        F_dff = F_dff_m
        SNR  = SNR_m
        cnn  = cnn_m
        rval = rval_m
        nsn  = nsn_m
        g    = g_m.T if g_m.ndim == 2 and g_m.shape[1] > 0 else np.zeros((1, n_main))
        idx_comp_bad = idx_bad

    return Estimates.from_arrays(
        A=A, dims=dims, contour_thr=contour_thr,
        C=C, YrA=YrA, S=S, F_dff=F_dff,
        SNR_comp=SNR, cnn_preds=cnn, r_values=rval,
        g=g,
        idx_components=idx_comp,
        idx_components_bad=idx_comp_bad,
        sn=sn, b=b, f=bg_f, neurons_sn=nsn,
        eval_params_caiman=eval_params,
        init_params_caiman=init_params,
        num_cells_original=A.shape[1],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_estimates_group(h5file, group_path: str):
    """Load arrays from one estimates group (main or discarded).

    Returns:
        Tuple of (A, C, YrA, S, F_dff, SNR_comp, cnn_preds, r_values,
                  g, neurons_sn, idx_components, idx_components_bad).
    """
    g = h5file[group_path]

    A    = _load_sparse_A(g["A"])
    C    = np.array(g["C"][:], dtype=np.float64)
    YrA  = np.array(g["YrA"][:], dtype=np.float64)
    S    = np.array(g["S"][:], dtype=np.float64)

    if _has_data(g, "F_dff"):
        F_dff = np.array(g["F_dff"][:], dtype=np.float64)
    else:
        F_dff = np.zeros_like(C)

    SNR_comp  = np.array(g["SNR_comp"][:],  dtype=np.float64)
    cnn_preds = np.array(g["cnn_preds"][:], dtype=np.float32)
    r_values  = np.array(g["r_values"][:],  dtype=np.float64)

    # g coefficients — discarded group often has shape (n, 0)
    g_raw = g["g"]
    g_arr = np.array(g_raw[:], dtype=np.float64) if g_raw.shape[1] > 0 else np.zeros((C.shape[0], 0))

    neurons_sn = _decode_s32_array(g["neurons_sn"][:]) if "neurons_sn" in g else np.zeros(C.shape[0])

    if "idx_components" in g and _has_data(g, "idx_components"):
        idx_comp = np.array(g["idx_components"][:], dtype=np.int32)
    else:
        idx_comp = np.array([], dtype=np.int32)

    if "idx_components_bad" in g and _has_data(g, "idx_components_bad"):
        idx_bad = np.array(g["idx_components_bad"][:], dtype=np.int32)
    else:
        idx_bad = np.array([], dtype=np.int32)

    return A, C, YrA, S, F_dff, SNR_comp, cnn_preds, r_values, g_arr, neurons_sn, idx_comp, idx_bad


def _load_sparse_A(group) -> csc_matrix:
    """Reconstruct scipy.sparse.csc_matrix from CaImAn HDF5 sparse group."""
    data    = np.array(group["data"][:])
    indices = np.array(group["indices"][:])
    indptr  = np.array(group["indptr"][:])
    shape   = tuple(group["shape"][:])           # (n_pixels, n_cells)
    return csc_matrix((data, indices, indptr), shape=shape)


def _compute_contours(A, dims: tuple[int, int], thr: float = 0.01) -> list | None:
    """Compute per-cell outer-boundary contours from spatial components."""
    try:
        from caiman_sorter_py.core.contours import compute_contours
        return compute_contours(A, dims, thr=thr)
    except Exception:
        return None


def _has_data(node, key: str) -> bool:
    """Return True if key exists in node and its dataset has non-zero shape."""
    try:
        ds = node[key]
        return len(ds.shape) > 0 and all(s > 0 for s in ds.shape)
    except (KeyError, AttributeError):
        return False


def _decode_s32_array(arr) -> np.ndarray:
    """Decode a |S32 byte-string array (CaImAn scalar serialisation) to float64."""
    result = np.full(len(arr), np.nan)
    for i, v in enumerate(arr):
        try:
            result[i] = float(v.decode() if isinstance(v, (bytes, np.bytes_)) else v)
        except (ValueError, TypeError, AttributeError):
            pass
    return result


def _load_init_params(f) -> dict:
    """Load the entire /params group as a nested dict.

    Mirrors the source CaImAn HDF5 layout exactly: subgroups (data, init,
    merging, motion, online, patch, preprocess, quality, ring_CNN, spatial,
    temporal) become nested dicts, datasets become scalars / arrays / strings.
    """
    if "params" not in f:
        return {}
    return _h5_group_to_dict(f["params"])


def _h5_group_to_dict(group) -> dict:
    """Recursively convert an h5py Group into a nested Python dict."""
    out: dict = {}
    for k in group.keys():
        item = group[k]
        if isinstance(item, h5py.Group):
            out[str(k)] = _h5_group_to_dict(item)
        else:
            out[str(k)] = _h5_dataset_to_value(item)
    for k, v in group.attrs.items():
        out[str(k)] = v.item() if isinstance(v, np.generic) else v
    return out


def _h5_dataset_to_value(ds):
    """Convert an h5py Dataset into a Python scalar, str, list, or ndarray.

    CaImAn serialises Python None / type names as |S* byte strings — we decode
    those and surface 'NoneType' as Python None so the dict round-trips cleanly.
    Embedded NUL bytes (from fixed-length |S32 padding) are stripped so the
    values can be re-saved through h5py's variable-length string type.
    """
    def _clean(s):
        if isinstance(s, (bytes, np.bytes_)):
            s = s.decode(errors="replace")
        if isinstance(s, str):
            return s.replace("\x00", "")
        return s

    shape = ds.shape
    if shape == ():
        v = ds[()]
        if isinstance(v, (bytes, np.bytes_, str)):
            s = _clean(v)
            return None if s == "NoneType" else s
        if isinstance(v, np.generic):
            return v.item()
        return v
    arr = ds[:]
    if arr.dtype.kind in ("S", "O", "U"):
        decoded = [_clean(b) for b in arr.ravel().tolist()]
        # Collapse single-element arrays to a bare scalar
        if len(decoded) == 1:
            s = decoded[0]
            return None if s == "NoneType" else s
        return decoded
    return np.asarray(arr)


def _load_eval_params(f) -> dict:
    key_map = {
        "params/quality/min_SNR":      "snr_thresh",
        "params/quality/SNR_lowest":   "snr_lowest_thresh",
        "params/quality/min_cnn_thr":  "cnn_thresh",
        "params/quality/cnn_lowest":   "cnn_lowest_thresh",
        "params/quality/rval_thr":     "rval_thresh",
        "params/quality/rval_lowest":  "rval_lowest_thresh",
    }
    return {
        name: float(f[hdf_key][()])
        for hdf_key, name in key_map.items()
        if hdf_key in f
    }
