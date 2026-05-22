"""Calcium trace plot panel."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)


class _MinimalToolbar(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ("Home", "Pan", "Zoom")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_home = None      # callable invoked AFTER home() completes

    def home(self, *args, **kwargs):
        super().home(*args, **kwargs)
        if self._on_home is not None:
            self._on_home()


# Trace definitions: name → (label, default color, default visible)
TRACES = {
    "raw":      ("Raw (C+YrA)",  "#d95f02", True),
    "denoised": ("Denoised (C)", "#1f78b4", True),
    "spikes":   ("Spikes (S)",   "#e6ab02", True),
    "dfdt":     ("Smooth dF/dt", "#e31a1c", False),
    "foopsi":   ("Foopsi (C)",   "#7b2d8b", False),
    "foopsi_s": ("Foopsi (S)",   "#1b7837", False),
}


class TracePanel(QWidget):
    """Calcium trace plot with per-trace toggle and scale/shift controls."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._lines: dict    = {}   # name → Line2D
        self._raw_data: dict = {}   # name → np.ndarray (unscaled)
        self._t: np.ndarray | None = None   # time axis in seconds (cached on data load)
        self._fr_cached: float | None = None  # sampling rate cached alongside _t
        self._last_cell: int | None = None  # last cell shown; reset xlim/ylim only when this changes
        # User-zoom state. When the user pans or zooms the trace plot, we stop
        # auto-rescaling on trace toggles so their view survives. Cleared by
        # cell change and by the toolbar Home button.
        self._user_zoomed: bool = False
        self._suppress_zoom_detect: bool = False
        self._build_ui()
        self._connect_session()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        controls = self._build_controls()
        controls.setMaximumWidth(290)
        layout.addWidget(controls)
        layout.addWidget(self._build_canvas(), stretch=1)

    def _build_canvas(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.fig = Figure(tight_layout=True, facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="gray")
        self.ax.spines["bottom"].set_color("gray")
        self.ax.spines["left"].set_color("gray")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.set_xlabel("Time (s)", color="gray")
        self.ax.set_ylabel("Fluorescence", color="gray")
        self.ax.text(0.5, 0.5, "No data loaded",
                     ha="center", va="center",
                     color="gray", transform=self.ax.transAxes)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = _MinimalToolbar(self.canvas, wrapper)
        self.toolbar.setMaximumHeight(28)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas)
        return wrapper

    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Traces")
        box.setToolTip(
            "Toggle traces shown in the plot for the current cell.\n"
            "Y-axis auto-scales to the union of visible traces."
        )
        grid = QGridLayout(box)
        grid.setSpacing(4)

        self._toggles: dict[str, QCheckBox] = {}

        grid.addWidget(QLabel("<b>Show</b>"), 0, 1)

        TRACE_TIPS = {
            "raw":      "C + YrA — raw fluorescence (denoised + residual).",
            "denoised": "C — CaImAn's denoised (AR-model-fit) calcium trace.",
            "spikes":   "S — CaImAn's deconvolved spike train (optionally smoothed via Spikes tab).",
            "dfdt":     "Smooth dF/dt — Gaussian-smoothed derivative of (C + YrA). Recomputed live from params.",
            "foopsi":   "Foopsi (C) — denoised calcium trace from constrained foopsi / OASIS.",
            "foopsi_s": "Foopsi (S) — deconvolved spike train from constrained foopsi / OASIS.",
        }
        for row, (name, (label, color, default_on)) in enumerate(TRACES.items(), start=1):
            swatch = QLabel(f"<span style='color:{color}'>■</span> {label}")
            tip = TRACE_TIPS.get(name, "")
            swatch.setToolTip(tip)
            grid.addWidget(swatch, row, 0)

            chk = QCheckBox()
            chk.setChecked(default_on)
            chk.setToolTip(tip)
            chk.stateChanged.connect(lambda state, n=name: self._on_toggle(n, bool(state)))
            grid.addWidget(chk, row, 1)
            self._toggles[name] = chk

        return box

    # ------------------------------------------------------------------
    # Session wiring
    # ------------------------------------------------------------------

    def _connect_session(self) -> None:
        self.session.add_listener("data_loaded", self._on_data_loaded)
        self.session.add_listener("cell_selected", self.show_cell)

    def _on_data_loaded(self) -> None:
        self._init_lines()
        self._last_cell = None   # force a full-view reset on first cell shown
        # Cache the time axis once — only depends on n_frames and fr, both of
        # which are fixed for a given session.
        from caiman_sorter_py.core.state import get_init_param
        est = self.session.est
        if est is not None:
            fr = float(get_init_param(est.init_params_caiman, "fr", 30))
            self._t = np.arange(est.C.shape[1]) / fr
            self._fr_cached = fr
        if self.session.proc:
            self.show_cell(self.session.current_cell)

    # ------------------------------------------------------------------
    # Core display methods
    # ------------------------------------------------------------------

    def _init_lines(self) -> None:
        """Create one persistent Line2D per trace. Called once after load."""
        self.ax.cla()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="gray")
        self.ax.spines["bottom"].set_color("gray")
        self.ax.spines["left"].set_color("gray")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.set_xlabel("Time (s)", color="gray")
        self.ax.set_ylabel("Fluorescence", color="gray")

        self._lines.clear()
        for name, (label, color, default_on) in TRACES.items():
            line, = self.ax.plot([], [], color=color, lw=0.8, label=label)
            line.set_visible(self._toggles[name].isChecked())
            self._lines[name] = line

        # Detect user pan/zoom by watching axis limit changes. Our own
        # programmatic limit changes (cell change, rescale) are wrapped in
        # _suppress_zoom_detect so they don't trip the flag.
        self.ax.callbacks.connect('xlim_changed', self._on_view_change)
        self.ax.callbacks.connect('ylim_changed', self._on_view_change)
        # Home button → clear the user-zoom flag so the next toggle re-fits.
        self.toolbar._on_home = lambda: setattr(self, '_user_zoomed', False)

        self.canvas.draw_idle()

    def _on_view_change(self, ax):
        if not self._suppress_zoom_detect:
            self._user_zoomed = True

    def show_cell(self, cell_idx: int) -> None:
        """Update all trace data for a new cell."""
        if not self._lines or self.session.est is None:
            return

        est  = self.session.est
        proc = self.session.proc
        # _t and fr are cached in _on_data_loaded; only recompute on the unusual
        # path where show_cell runs before that listener fired.
        n_frames = est.C.shape[1]
        if self._t is None or len(self._t) != n_frames:
            from caiman_sorter_py.core.state import get_init_param
            fr = float(get_init_param(est.init_params_caiman, "fr", 30))
            self._t = np.arange(n_frames) / fr
            self._fr_cached = fr
        fr = self._fr_cached

        def _deconv_get(lst):
            try:
                v = lst[cell_idx]
                return np.asarray(v, dtype=float) if v is not None else np.zeros(n_frames)
            except (IndexError, TypeError):
                return np.zeros(n_frames)

        from scipy.ndimage import gaussian_filter1d
        dt_ms = 1000.0 / fr

        def _smooth(arr, sigma_ms):
            sigma_frames = max(sigma_ms / dt_ms, 0.5)
            return gaussian_filter1d(arr.astype(float),
                                     sigma=sigma_frames, mode="reflect")

        sp_ops = self.session.ops.spikes
        spikes = est.S[cell_idx].copy()
        if sp_ops.smooth and sp_ops.smooth_sigma > 0:
            spikes = _smooth(spikes, sp_ops.smooth_sigma)

        foopsi_s = _deconv_get(proc.foopsi.S)
        fp_ops = self.session.ops.foopsi
        if fp_ops.smooth_s and fp_ops.smooth_sigma > 0:
            foopsi_s = _smooth(foopsi_s, fp_ops.smooth_sigma)

        # smooth dF/dt: compute live from the current params (single cell only).
        # The stored proc.smooth_dfdt.S is only updated when the user clicks the
        # "Run smooth dF/dt" button (for all cells); display always reflects the
        # live params, even before any Run.
        from caiman_sorter_py.core.deconvolution import (
            apply_smooth_dfdt_threshold, compute_smooth_dfdt,
        )
        single = (est.C[cell_idx:cell_idx + 1] + est.YrA[cell_idx:cell_idx + 1])
        dfdt = compute_smooth_dfdt(single, fr, self.session.ops.smooth_dfdt)
        dfdt = apply_smooth_dfdt_threshold(dfdt, self.session.ops.smooth_dfdt)[0]

        self._raw_data = {
            "raw":      est.C[cell_idx] + est.YrA[cell_idx],
            "denoised": est.C[cell_idx].copy(),
            "spikes":   spikes,
            "dfdt":     dfdt,
            "foopsi":   _deconv_get(proc.foopsi.C),
            "foopsi_s": foopsi_s,
        }

        for name, line in self._lines.items():
            scale, shift = self._get_scale_shift(name)
            line.set_data(self._t, self._raw_data[name] * scale + shift)

        if cell_idx != self._last_cell:
            # New cell: reset view to full range and clear the user-zoom
            # flag — the previous cell's zoom doesn't apply to this one.
            self._suppress_zoom_detect = True
            try:
                self.ax.set_xlim(0, self._t[-1])
                self._rescale_y(force=True)
            finally:
                self._suppress_zoom_detect = False
            self._user_zoomed = False
            self._last_cell = cell_idx
        else:
            # Same cell, data updated (e.g. deconv re-run or scale/shift):
            # preserve current zoom — just redraw with new line data.
            self.canvas.draw_idle()

    def set_trace_visible(self, name: str, visible: bool) -> None:
        """Show or hide a named trace without replotting.

        Auto-rescales y to the union of visible traces — UNLESS the user has
        zoomed/panned the plot, in which case we preserve their view. They
        can reset by clicking Home on the toolbar.
        """
        if name in self._lines:
            self._lines[name].set_visible(visible)
            if self._user_zoomed:
                self.canvas.draw_idle()
            else:
                self._rescale_y()

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def _on_toggle(self, name: str, on: bool) -> None:
        if self._lines:
            self.set_trace_visible(name, on)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scale_shift(self, name: str) -> tuple[float, float]:
        """Return (scale, shift) for a trace.

        raw/denoised always use (1, 0). foopsi C uses (1, 0); foopsi S
        and spikes have their own scale/shift; smooth dF/dt has its own.
        """
        ops = self.session.ops
        if name == "spikes":
            return ops.spikes.scale, ops.spikes.shift
        if name == "dfdt":
            return ops.smooth_dfdt.scale, ops.smooth_dfdt.shift
        if name == "foopsi_s":
            return ops.foopsi.scale, ops.foopsi.shift
        return 1.0, 0.0

    def _rescale_y(self, force: bool = False) -> None:
        """Fit y-axis to the union of all currently visible traces.

        Wraps the limit change in `_suppress_zoom_detect` so the change isn't
        mistaken for a user pan/zoom. `force=True` lets callers bypass the
        user-zoom guard (used internally on cell change).
        """
        if self._user_zoomed and not force:
            self.canvas.draw_idle()
            return

        ymin, ymax = np.inf, -np.inf
        for name, line in self._lines.items():
            if not line.get_visible() or name not in self._raw_data:
                continue
            scale, shift = self._get_scale_shift(name)
            y = self._raw_data[name] * scale + shift
            if len(y):
                ymin = min(ymin, float(np.nanmin(y)))
                ymax = max(ymax, float(np.nanmax(y)))

        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = (ymax - ymin) * 0.05
            prev_suppress = self._suppress_zoom_detect
            self._suppress_zoom_detect = True
            try:
                self.ax.set_ylim(ymin - pad, ymax + pad)
            finally:
                self._suppress_zoom_detect = prev_suppress

        self.canvas.draw_idle()
