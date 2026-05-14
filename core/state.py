"""Core data structures and the Session state object.

Defines:
  - Estimates: raw CaImAn output (A, C, YrA, S, metrics, contours, ...)
  - Proc: derived processing results (noise, AR coeffs, peaks, accepted mask, ...)
  - Ops: all GUI parameters (eval thresholds, deconv params, file paths, ...)
  - Session: QObject that owns est/proc/ops and emits signals on state changes.
             Single source of truth for the entire application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix


# Short-name → nested CaImAn params path (mirrors source HDF5 /params group).
_INIT_PARAM_PATHS = {
    "fr":           ("data", "fr"),
    "ar_order":     ("preprocess", "p"),
    "fudge_factor": ("temporal", "fudge_factor"),
    "lags":         ("temporal", "lags"),
    "gSig":         ("init", "gSig"),
}


def get_init_param(init: Optional[dict], key: str, default=None):
    """Look up a CaImAn init param by short name, against either layout.

    Newer loads store the full nested CaImAn params struct (data/init/temporal/...);
    legacy session files may have flat keys at the top level. Try the flat top
    level first, then fall back to the nested path.
    """
    if not init:
        return default
    if key in init and not isinstance(init[key], dict):
        return init[key]
    path = _INIT_PARAM_PATHS.get(key)
    if path is None:
        return default
    cur = init
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@dataclass
class Estimates:
    """Raw estimates loaded from a CaImAn HDF5 file."""
    A: csc_matrix                        # (n_pixels, n_cells) spatial components
    C: np.ndarray                        # (n_cells, n_frames) denoised traces
    YrA: np.ndarray                      # (n_cells, n_frames) residual traces
    S: np.ndarray                        # (n_cells, n_frames) spike trains
    F_dff: np.ndarray                    # (n_cells, n_frames) ΔF/F traces
    SNR_comp: np.ndarray                 # (n_cells,) CaImAn SNR
    cnn_preds: np.ndarray                # (n_cells,) CNN classifier probability
    r_values: np.ndarray                 # (n_cells,) spatial correlation
    g: np.ndarray                        # (2, n_cells) AR model coefficients
    dims: tuple[int, int]                # (height, width) of the FOV
    idx_components: np.ndarray           # CaImAn-accepted cell indices (0-based)
    idx_components_bad: np.ndarray       # CaImAn-rejected cell indices (0-based)
    contours: Optional[list] = None      # list of dicts from get_contours()
    sn: Optional[np.ndarray] = None      # (n_pixels,) pixel noise
    neurons_sn: Optional[np.ndarray] = None  # (n_cells,) per-neuron noise from CaImAn
    b: Optional[np.ndarray] = None       # spatial background
    f: Optional[np.ndarray] = None       # temporal background
    eval_params_caiman: Optional[dict] = None   # thresholds used by CaImAn
    init_params_caiman: Optional[dict] = None   # data/init params from CaImAn
    num_cells_original: int = 0          # set after load; used for reset


@dataclass
class DeconvResults:
    """Deconvolution results for one method, stored per cell."""
    S: list = field(default_factory=list)    # spike trains, one array per cell
    C: list = field(default_factory=list)    # denoised traces, one per cell
    g: list = field(default_factory=list)    # AR coefficients, one per cell


@dataclass
class Proc:
    """Derived processing results computed from Estimates."""
    # Cell dimensions
    num_cells: int = 0
    num_frames: int = 0

    # Acceptance state — single source of truth
    accepted: Optional[np.ndarray] = None        # (n_cells,) bool, current state
    accepted_core: Optional[np.ndarray] = None   # (n_cells,) bool, auto-eval only
    manual_override: Optional[np.ndarray] = None # (n_cells,) bool, manually edited

    # Per-cell metrics
    noise: Optional[np.ndarray] = None           # (n_cells,) noise std
    skewness: Optional[np.ndarray] = None        # (n_cells,)
    peaks_ave: Optional[np.ndarray] = None       # (n_cells,) average of top-N peaks
    num_zeros: Optional[np.ndarray] = None       # (n_cells,) zero-count in YrA
    SNR2_vals: Optional[np.ndarray] = None       # (n_cells,) peaks_ave / noise
    firing_stab_vals: Optional[np.ndarray] = None # (n_cells,) firing stability

    # AR model time constants
    gAR1: Optional[np.ndarray] = None            # (n_cells,) AR(1) coefficient
    gAR2: Optional[np.ndarray] = None            # (n_cells, 2) AR(2) coefficients
    tauAR1: Optional[np.ndarray] = None          # (n_cells,) decay time constant
    tauAR2: Optional[np.ndarray] = None          # (n_cells, 2) rise/decay constants

    # Deconvolution outputs (populated on demand)
    smooth_dfdt: DeconvResults = field(default_factory=DeconvResults)
    smooth_dfdt_std: Optional[np.ndarray] = None  # (n_cells,) std of smooth_dfdt.S
    foopsi: DeconvResults = field(default_factory=DeconvResults)


@dataclass
class EvalParamsCaiman:
    """Thresholds for the CaImAn-style evaluation method."""
    snr_thresh: float = 2.0
    snr_lowest_thresh: float = 0.5
    cnn_thresh: float = 0.99
    cnn_lowest_thresh: float = 0.1
    rval_thresh: float = 0.8
    rval_lowest_thresh: float = -1.0


@dataclass
class EvalParamsReject:
    """Thresholds and toggles for the reject-threshold evaluation method."""
    use_snr_caiman: bool = False
    snr_caiman: float = 2.0
    use_snr2: bool = False
    snr2: float = 2.0
    use_cnn: bool = False
    cnn: float = 0.5
    use_rvalues: bool = False
    rvalues: float = 0.5
    use_min_sig_frac: bool = False
    min_sig_frac: float = 0.1
    use_firing_stability: bool = False
    firing_stability: float = 0.0
    use_skewness: bool = False
    skewness: float = 0.0


@dataclass
class DeconvParams:
    """Parameters for one deconvolution method."""
    ar_order: int = 1                    # AR(1) or AR(2)
    manual_tau: bool = False             # use tau_decay/tau_rise instead of cached gAR
    tau_decay: float = 0.4               # seconds (used when manual_tau)
    tau_rise: float = 0.1                # seconds (AR2 only, used when manual_tau)
    fudge_factor: float = 0.99           # AR coefficient bias correction
    solver: str = "oasis"                # 'oasis' | 'cvxpy' | 'cvx'
    scale: float = 1.0                   # display-only scale
    shift: float = 0.0                   # display-only shift
    smooth_s: bool = False               # display-only Gaussian smoothing of spike trace
    smooth_sigma: float = 50.0           # smoothing sigma in ms (when smooth_s)


@dataclass
class MergeParams:
    """Parameters for the duplicate-component detection step.

    Mirrors MATLAB f_cs_find_similar_comp.m — spatial overlap thresholding
    on A^T A, followed by temporal correlation of (C + YrA).
    """
    method: str = "choose best snr"      # one of: 'choose best snr', 'weighted ave',
                                         # 'full mean corr', 'svd', 'nmf'
    spatial_thr: float = 0.5             # overlap threshold (A^T A value)
    temporal_thr: float = 0.5            # Pearson correlation threshold
    use_accepted_only: bool = True       # only compare currently accepted cells


@dataclass
class SpikesParams:
    """Display-only post-processing for the raw CaImAn spike trace (est.S)."""
    smooth: bool = False
    smooth_sigma: float = 50.0           # ms
    scale: float = 1.0
    shift: float = 0.0


@dataclass
class SmoothDfdtParams:
    """Parameters for the smooth dF/dt (Gaussian convolution) method."""
    gauss_sigma: float = 50.0            # ms
    rectify: bool = False
    normalize: bool = False
    apply_thresh: bool = False
    threshold_z: float = 1.0
    scale: float = 1.0
    shift: float = 0.0
    plot_threshold: bool = False


@dataclass
class Ops:
    """All GUI parameters: evaluation thresholds, deconv params, file paths."""
    eval_method: str = "caiman"          # "caiman" or "reject_threshold"
    eval_caiman: EvalParamsCaiman = field(default_factory=EvalParamsCaiman)
    eval_reject: EvalParamsReject = field(default_factory=EvalParamsReject)
    spikes: SpikesParams = field(default_factory=SpikesParams)
    smooth_dfdt: SmoothDfdtParams = field(default_factory=SmoothDfdtParams)
    foopsi: DeconvParams = field(default_factory=DeconvParams)
    merge: MergeParams = field(default_factory=MergeParams)
    load_caiman_rejected: bool = False   # include CaImAn-rejected components on load
    save_tag: str = "_sort"              # string appended to source stem in default save names
    save_as_mat: bool = False            # also write a MATLAB-compatible .mat alongside the .h5 save
    contour_thr: float = 0.01            # amplitude threshold for contour tracing (fraction of peak)
    browse_path: str = ""
    ops_path: str = ""


class Session:
    """Central state object. Owns est, proc, ops.

    UI panels should read from and write to this object only.
    Changes are communicated via callbacks registered with add_listener().

    Signals (as plain callbacks for now; can be upgraded to pyqtSignal later):
        on_cell_selected(cell_idx: int)
        on_cell_accepted_changed(cell_idx: int)
        on_cells_reevaluated()
        on_data_loaded()
    """

    def __init__(self):
        self.est: Optional[Estimates] = None
        self.proc: Optional[Proc] = None
        self.ops: Ops = Ops()
        self.current_cell: int = 0
        self._listeners: dict[str, list] = {
            "cell_selected": [],
            "cell_accepted_changed": [],
            "cells_reevaluated": [],
            "data_loaded": [],
        }

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------

    def add_listener(self, event: str, callback) -> None:
        """Register a callback for a named event."""
        self._listeners[event].append(callback)

    def _emit(self, event: str, *args) -> None:
        for cb in self._listeners[event]:
            cb(*args)

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def load_data(self, est: Estimates, proc: Proc) -> None:
        """Set new est and proc after a file load. Resets current cell."""
        self.est = est
        self.proc = proc
        self.current_cell = 0
        self._emit("data_loaded")

    def select_cell(self, idx: int) -> None:
        """Change the currently displayed cell."""
        if idx == self.current_cell:
            return
        self.current_cell = idx
        self._emit("cell_selected", idx)

    def set_accepted(self, idx: int, value: bool) -> None:
        """Accept or reject a single cell. Marks it as manually overridden."""
        if self.proc.accepted[idx] == value:
            return
        self.proc.accepted[idx] = value
        self.proc.manual_override[idx] = True
        self._emit("cell_accepted_changed", idx)

    def reevaluate_all(self) -> None:
        """Re-run automatic evaluation and emit a bulk change signal."""
        self._emit("cells_reevaluated")

    def refresh_cell(self) -> None:
        """Re-emit cell_selected for the current cell (e.g. after deconv param change)."""
        self._emit("cell_selected", self.current_cell)
