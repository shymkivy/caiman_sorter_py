"""Accepted and rejected component image panels."""
from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)


class _MinimalToolbar(NavigationToolbar2QT):
    """Navigation toolbar with only Home, Pan, and Zoom buttons."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ("Home", "Pan", "Zoom")]


class ImagePanel(QWidget):
    """Two side-by-side matplotlib images (accepted / rejected) with contour controls."""

    METRICS  = ["None", "SNR (CaImAn)", "SNR2", "CNN", "R values", "Firing stability"]
    BKG_MODES = ["Components", "Weighted comp", "W comp + bkg"]
    CMAP     = "viridis"

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._contour_lines: list = []
        self._accepted_im = None
        self._rejected_im = None
        self._build_ui()
        self._connect_session()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        layout.addLayout(self._build_controls())
        self._canvas_splitter = QSplitter(Qt.Horizontal)
        self._canvas_splitter.addWidget(self._build_accepted_canvas())
        self._canvas_splitter.addWidget(self._build_rejected_canvas())
        self._canvas_splitter.setSizes([1, 1])
        layout.addWidget(self._canvas_splitter, stretch=1)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Contours:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.METRICS)
        self.metric_combo.setToolTip(
            "Color each cell's contour by a per-cell metric.\n"
            "  • None  — single color (no colorbar)\n"
            "  • SNR / CNN / r-values — CaImAn-derived quality metrics\n"
            "  • SNR2 / firing_stab / noise / skewness / peaks_ave — derived in proc init"
        )
        row.addWidget(self.metric_combo)

        # Colorbar — shown next to the dropdown when a metric is active
        self.cbar_fig    = Figure(figsize=(2, 0.28), facecolor="#1e1e1e")
        self.cbar_ax     = self.cbar_fig.add_axes([0.04, 0.38, 0.92, 0.52])
        self.cbar_canvas = FigureCanvasQTAgg(self.cbar_fig)
        self.cbar_canvas.setFixedSize(170, 28)
        self.cbar_canvas.setVisible(False)
        self.cbar_canvas.setToolTip(
            "Color scale for the selected metric. Auto-ranges to the 0.5–99.5 percentile\n"
            "across cells so outliers don't dominate."
        )
        row.addWidget(self.cbar_canvas)

        row.addWidget(QLabel("Background:"))
        self.bkg_combo = QComboBox()
        self.bkg_combo.addItems(self.BKG_MODES)
        self.bkg_combo.setToolTip(
            "Background image behind the contours.\n"
            "  • Max projection      — bright pixels = strong components\n"
            "  • Mean image          — average activity\n"
            "  • Correlation image   — spatially correlated activity (slower to compute)\n"
            "  • Off                 — black background only"
        )
        row.addWidget(self.bkg_combo)
        row.addStretch()
        return row

    def _build_accepted_canvas(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self.accepted_fig = Figure(facecolor="#1e1e1e")
        self.accepted_ax  = self.accepted_fig.add_axes([0, 0, 1, 1])
        self._style_ax(self.accepted_ax)
        self.accepted_ax.text(0.5, 0.5, "No data loaded",
                              ha="center", va="center",
                              color="gray", transform=self.accepted_ax.transAxes)
        self.accepted_canvas = FigureCanvasQTAgg(self.accepted_fig)
        self.accepted_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.accepted_toolbar = _MinimalToolbar(self.accepted_canvas, wrapper)
        self.accepted_toolbar.setMaximumHeight(28)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.accepted_label = QLabel("Accepted (0)")
        self.accepted_label.setToolTip(
            "Composite footprint of all currently accepted cells.\n"
            "Click a contour to jump to that cell."
        )
        header.addWidget(self.accepted_label)
        header.addStretch()
        header.addWidget(self.accepted_toolbar)
        v.addLayout(header)
        v.addWidget(self.accepted_canvas)
        return wrapper

    def _build_rejected_canvas(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self.rejected_fig = Figure(facecolor="#1e1e1e")
        self.rejected_ax  = self.rejected_fig.add_axes([0, 0, 1, 1])
        self._style_ax(self.rejected_ax)
        self.rejected_ax.text(0.5, 0.5, "No data loaded",
                              ha="center", va="center",
                              color="gray", transform=self.rejected_ax.transAxes)
        self.rejected_canvas = FigureCanvasQTAgg(self.rejected_fig)
        self.rejected_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.rejected_toolbar = _MinimalToolbar(self.rejected_canvas, wrapper)
        self.rejected_toolbar.setMaximumHeight(28)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.rejected_label = QLabel("Rejected (0)")
        self.rejected_label.setToolTip(
            "Composite footprint of all currently rejected cells.\n"
            "Click a contour to jump to that cell."
        )
        header.addWidget(self.rejected_label)
        header.addStretch()
        header.addWidget(self.rejected_toolbar)
        v.addLayout(header)
        v.addWidget(self.rejected_canvas)
        return wrapper

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------
    # Session wiring
    # ------------------------------------------------------------------

    def _connect_session(self) -> None:
        self.session.add_listener("data_loaded", self.refresh_images)
        self.session.add_listener("cell_selected", self.highlight_cell)
        self.session.add_listener("cell_accepted_changed", self.update_cell_toggle)
        self.session.add_listener("cells_reevaluated", self.refresh_images)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)
        self.bkg_combo.currentTextChanged.connect(self._on_bkg_changed)
        self.accepted_canvas.mpl_connect("button_press_event", self._on_click)
        self.rejected_canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def refresh_images(self) -> None:
        """Rebuild both composite images from scratch."""
        if self.session.est is None:
            return
        self._clear_axes()
        self._draw_backgrounds()
        self._draw_all_contours()
        self._update_labels()
        if self.session.proc is not None:
            self.highlight_cell(self.session.current_cell)
        else:
            self.accepted_canvas.draw_idle()
            self.rejected_canvas.draw_idle()

    def update_cell_toggle(self, cell_idx: int) -> None:
        self.refresh_images()

    def highlight_cell(self, cell_idx: int) -> None:
        """Highlight the current cell's contour; dim all others."""
        if not self._contour_lines:
            return
        for i, line in enumerate(self._contour_lines):
            if line is None:
                continue
            if i == cell_idx:
                line.set_linewidth(3.5)
                line.set_zorder(10)
                line.set_alpha(1.0)
            else:
                line.set_linewidth(1.2)
                line.set_zorder(1)
                line.set_alpha(0.6)
        self.accepted_canvas.draw_idle()
        self.rejected_canvas.draw_idle()

    def set_contour_metric(self, metric: str) -> None:
        """Recolor all contours by the given metric."""
        if not self._contour_lines or self.session.proc is None:
            return
        vals        = None if metric == "None" else self._get_metric_array(metric)
        color_range = self._metric_percentile_range(vals)
        for cell_idx, line in enumerate(self._contour_lines):
            if line is None:
                continue
            line.set_color(self._cell_color(cell_idx, vals, color_range))
        self._update_colorbar(vals, color_range)
        self.accepted_canvas.draw_idle()
        self.rejected_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_axes(self) -> None:
        for ax in (self.accepted_ax, self.rejected_ax):
            ax.cla()
            ax.set_position([0, 0, 1, 1])   # restore after cla() resets it
            self._style_ax(ax)

    def _build_bg_image(self, mask: np.ndarray) -> np.ndarray:
        est      = self.session.est
        dims     = est.dims
        bkg_mode = self.bkg_combo.currentText()
        if not mask.any():
            return np.zeros(dims)
        A_sub = est.A[:, mask]
        if bkg_mode == "Components":
            img = np.asarray(A_sub.sum(axis=1)).ravel().reshape(dims, order="F")
        else:
            col_max = np.asarray(A_sub.max(axis=0).toarray()).ravel()
            col_max[col_max == 0] = 1.0
            weights = 1.0 / col_max
            img = np.asarray(A_sub @ weights).ravel().reshape(dims, order="F")
            if bkg_mode == "W comp + bkg" and est.b is not None:
                b = np.asarray(est.b)
                b_img = (b.sum(axis=1) if b.ndim == 2 else b).ravel().reshape(dims, order="F")
                img = img + b_img * 0.2
        return img

    def _draw_backgrounds(self) -> None:
        proc    = self.session.proc
        acc_img = self._build_bg_image(proc.accepted)
        rej_img = self._build_bg_image(~proc.accepted)

        self._accepted_im = self.accepted_ax.imshow(
            acc_img, cmap=self.CMAP, aspect="equal", origin="lower",
            interpolation="nearest", **self._clim(acc_img),
        )
        self._rejected_im = self.rejected_ax.imshow(
            rej_img, cmap=self.CMAP, aspect="equal", origin="lower",
            interpolation="nearest", **self._clim(rej_img),
        )

    @staticmethod
    def _clim(img: np.ndarray, pct_lo: float = 0.5, pct_hi: float = 99.5) -> dict:
        """Return vmin/vmax based on percentiles of non-zero pixels."""
        nonzero = img[img > 0]
        if len(nonzero) == 0:
            return {"vmin": 0, "vmax": 1e-9}
        return {
            "vmin": float(np.percentile(nonzero, pct_lo)),
            "vmax": float(np.percentile(nonzero, pct_hi)),
        }

    def _update_colorbar(self, vals, color_range) -> None:
        """Refresh the colorbar next to the metric dropdown."""
        import matplotlib as mpl
        if vals is None or color_range is None:
            self.cbar_canvas.setVisible(False)
            return
        lo, hi = color_range
        self.cbar_ax.cla()
        norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
        cb = self.cbar_fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdYlGn),
            cax=self.cbar_ax,
            orientation="horizontal",
        )
        cb.ax.tick_params(colors="gray", labelsize=7, length=2, pad=1)
        for spine in cb.ax.spines.values():
            spine.set_edgecolor("gray")
        self.cbar_canvas.setVisible(True)
        self.cbar_canvas.draw_idle()

    def _get_metric_array(self, metric_text: str):
        est  = self.session.est
        proc = self.session.proc
        return {
            "SNR (CaImAn)":     est.SNR_comp,
            "SNR2":             proc.SNR2_vals,
            "CNN":              est.cnn_preds,
            "R values":         est.r_values,
            "Firing stability": proc.firing_stab_vals,
        }.get(metric_text)

    def _metric_percentile_range(self, vals):
        if vals is None:
            return None
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return None
        return float(np.percentile(finite, 5)), float(np.percentile(finite, 95))

    def _cell_color(self, cell_idx: int, vals, color_range):
        if vals is None or color_range is None:
            return "#00cc44" if self.session.proc.accepted[cell_idx] else "#cc3333"
        v = float(vals[cell_idx])
        if not np.isfinite(v):
            return "gray"
        lo, hi = color_range
        t = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        r, g, b, _ = cm.RdYlGn(t)
        return (r, g, b)

    def _draw_all_contours(self) -> None:
        est  = self.session.est
        proc = self.session.proc
        self._contour_lines.clear()

        if est.contours is None:
            return

        metric_text = self.metric_combo.currentText()
        vals        = None if metric_text == "None" else self._get_metric_array(metric_text)
        color_range = self._metric_percentile_range(vals)

        for cell_idx in range(proc.num_cells):
            contour = est.contours[cell_idx] if cell_idx < len(est.contours) else {}
            coords  = contour.get("coordinates") if isinstance(contour, dict) else None
            if coords is None or not hasattr(coords, "__len__") or len(coords) == 0:
                self._contour_lines.append(None)
                continue

            ax    = self.accepted_ax if proc.accepted[cell_idx] else self.rejected_ax
            color = self._cell_color(cell_idx, vals, color_range)
            line, = ax.plot(
                coords[:, 0], coords[:, 1],
                color=color, lw=1.5, alpha=0.7, solid_capstyle="round",
            )
            self._contour_lines.append(line)

        self._update_colorbar(vals, color_range)

    def _update_labels(self) -> None:
        proc  = self.session.proc
        n_acc = int(proc.accepted.sum())
        n_tot = proc.num_cells
        self.accepted_label.setText(f"Accepted ({n_acc})")
        self.rejected_label.setText(f"Rejected ({n_tot - n_acc})")

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_metric_changed(self, text: str) -> None:
        if self.session.est is not None:
            self.set_contour_metric(text)

    def _on_bkg_changed(self, text: str) -> None:
        if self.session.est is not None:
            self.refresh_images()

    def resizeEvent(self, event) -> None:
        """Keep each image approximately square (canvas height ≈ panel width / 2)."""
        super().resizeEvent(event)
        if not hasattr(self, '_canvas_splitter'):
            return
        target_h = max(120, self.width() // 2)
        if abs(self._canvas_splitter.maximumHeight() - target_h) > 4:
            self._canvas_splitter.setMaximumHeight(target_h)
            self._canvas_splitter.setMinimumHeight(max(100, target_h - 8))

    def _on_pick(self, event) -> None:
        pass

    def _on_click(self, event) -> None:
        """Select or toggle the cell at the clicked pixel.

        Ignored when either toolbar is in zoom/pan mode.
        Right-click toggles accept/reject; left-click selects.
        """
        # Yield to toolbar zoom/pan modes
        if self.accepted_toolbar.mode or self.rejected_toolbar.mode:
            return

        if self.session.est is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not (np.isfinite(event.xdata) and np.isfinite(event.ydata)):
            return
        if event.inaxes not in (self.accepted_ax, self.rejected_ax):
            return

        est    = self.session.est
        proc   = self.session.proc
        height, width = est.dims

        col = int(np.clip(round(event.xdata), 0, width  - 1))
        row = int(np.clip(round(event.ydata), 0, height - 1))

        if event.inaxes is self.accepted_ax:
            candidates = np.where(proc.accepted)[0]
        else:
            candidates = np.where(~proc.accepted)[0]

        if len(candidates) == 0:
            return

        linear_px = row + col * height
        A_row     = np.asarray(est.A.getrow(linear_px).todense()).ravel()
        pix_vals  = A_row[candidates]

        if len(pix_vals) > 0 and pix_vals.max() > 0:
            best = int(candidates[int(np.argmax(pix_vals))])
        else:
            best = self._nearest_com(candidates, row, col)

        if best is None:
            return

        if event.button == 3:
            self.session.set_accepted(best, not proc.accepted[best])
        else:
            self.session.select_cell(best)

    def _nearest_com(self, candidates: np.ndarray, row: int, col: int):
        est = self.session.est
        if est.contours is None:
            return int(candidates[0]) if len(candidates) else None
        best_idx, best_dist = None, np.inf
        for ci in candidates:
            c   = est.contours[ci] if ci < len(est.contours) else {}
            com = c.get("CoM") if isinstance(c, dict) else None
            if com is None:
                continue
            dist = (float(com[0]) - row) ** 2 + (float(com[1]) - col) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx  = int(ci)
        return best_idx
