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
        # Cached MATLAB-equivalent background pieces:
        #   _bkg_comp_weights : (n_cells,) per-cell mean denoised activity
        #   _bkg_bg_img       : (h, w)   spatial-background contribution (b @ mean(f))
        # Both are computed once on data_loaded and reused for every refresh.
        self._bkg_comp_weights = None
        self._bkg_bg_img       = None
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
            "  • None  — accepted cells green, rejected red (no colorbar).\n"
            "  • SNR (CaImAn) / CNN / R values — CaImAn-derived quality metrics.\n"
            "  • SNR2 / Firing stability — derived during proc init (peaks_ave/noise,\n"
            "    and a peak-rate firing-stability score).\n"
            "Color scale auto-ranges to the 0.5–99.5 percentile of finite values."
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
            "  • Components    — sum of footprints, sum(A[:, cells], axis=1).\n"
            "                    Every cell appears equally bright; useful for ROI placement.\n"
            "  • Weighted comp — |A[:, cells] @ mean(C)| — each cell scaled by its mean\n"
            "                    activity; the absolute value is taken so cells stay\n"
            "                    bright regardless of baseline sign (CaImAn-Python ships\n"
            "                    baseline-subtracted C).\n"
            "  • W comp + bkg  — weighted comp + CaImAn's spatial background\n"
            "                    (b @ mean(f)) — shows components on top of the\n"
            "                    structural FOV; color range clipped to 1–99.5%."
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
            "  • Left-click a contour to jump to that cell.\n"
            "  • Right-click a contour to toggle its accept/reject state\n"
            "    (also flags the cell as manually overridden).\n"
            "Clicks are ignored while Pan/Zoom is active in the toolbar."
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
            "  • Left-click a contour to jump to that cell.\n"
            "  • Right-click a contour to toggle its accept/reject state\n"
            "    (also flags the cell as manually overridden).\n"
            "Clicks are ignored while Pan/Zoom is active in the toolbar."
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
        self.session.add_listener("data_loaded", self._on_data_loaded)
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

    def _on_data_loaded(self) -> None:
        """Invalidate the bkg cache so it's rebuilt for the new dataset."""
        self._bkg_comp_weights = None
        self._bkg_bg_img       = None
        self.refresh_images()

    def refresh_images(self) -> None:
        """Rebuild both composite images from scratch."""
        if self.session.est is None:
            return
        self._ensure_bkg_cache()
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

    def _ensure_bkg_cache(self) -> None:
        """Pre-compute per-cell weights and the spatial-background image once.

        Mirrors MATLAB f_cs_initialize_GUI_params.m:
            bkg_comp_weights = mean(est.C, 2)               % (n_cells, 1)
            bkg_bgkcomp      = reshape(mean(est.f) * est.b, dims)
        """
        est = self.session.est
        if est is None:
            return
        n_cells = est.C.shape[0]
        if self._bkg_comp_weights is None or len(self._bkg_comp_weights) != n_cells:
            self._bkg_comp_weights = np.mean(est.C, axis=1)
        if self._bkg_bg_img is None:
            self._bkg_bg_img = self._compute_bg_component_image(est)

    @staticmethod
    def _compute_bg_component_image(est) -> np.ndarray | None:
        """Return CaImAn's spatial-background contribution as a (h, w) image.

            img = (b @ mean(f, axis=1)).reshape(dims, order='F')
        Returns None when b or f are absent.
        """
        if est.b is None or est.f is None or not est.dims:
            return None
        b = np.asarray(est.b)
        f = np.asarray(est.f)
        # CaImAn convention: b is (n_pixels, n_bg), f is (n_bg, n_frames)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if b.shape[1] != f.shape[0]:
            # Tolerate transposed b that some sources save
            if b.shape[0] == f.shape[0]:
                b = b.T
            else:
                return None
        mean_f = f.mean(axis=1)            # (n_bg,)
        flat   = b @ mean_f                # (n_pixels,)
        return np.asarray(flat).ravel().reshape(est.dims, order="F")

    def _build_bg_image(self, mask: np.ndarray) -> np.ndarray:
        est      = self.session.est
        dims     = est.dims
        bkg_mode = self.bkg_combo.currentText()
        if not mask.any():
            base = np.zeros(dims)
            if bkg_mode == "W comp + bkg" and self._bkg_bg_img is not None:
                base = base + self._bkg_bg_img
            return base

        A_sub = est.A[:, mask]
        if bkg_mode == "Components":
            img = np.asarray(A_sub.sum(axis=1)).ravel().reshape(dims, order="F")
        else:
            weights = np.asarray(self._bkg_comp_weights)[mask]
            img = np.asarray(A_sub @ weights).ravel().reshape(dims, order="F")
            if bkg_mode == "W comp + bkg" and self._bkg_bg_img is not None:
                # Literal MATLAB formula: weighted comp + spatial-background image.
                # `img` may be mixed-sign here (Python CaImAn ships baseline-
                # subtracted C); that's intentional — the bkg dominates and the
                # signed contribution modulates on top of it.
                img = img + self._bkg_bg_img
            elif bkg_mode == "Weighted comp":
                # Display magnitude so all active cells appear as bright spots on
                # a dark background instead of the (technically MATLAB-correct
                # but visually inverted) mid-grey background with cells dipping
                # below it.
                img = np.abs(img)
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

    def _clim(self, img: np.ndarray) -> dict:
        """Color range for the composite background.

        - "Components" / "Weighted comp": non-negative, mostly-zero images.
          Anchor at 0 (empty FOV reads as dark) and cap at 99.5-pct of non-zero
          pixels so a single hot footprint pixel doesn't compress the range.
        - "W comp + bkg": dominated by the `b @ mean(f)` background. A 1-pct
          vmin trims the lower tail so the bkg pixels are mid-colormap rather
          than all crushed to dark, then 99.5-pct vmax leaves room for cells
          to pop on top.
        """
        if img.size == 0:
            return {"vmin": 0.0, "vmax": 1e-9}
        bkg_mode = self.bkg_combo.currentText()
        if bkg_mode == "W comp + bkg":
            lo = float(np.percentile(img, 1.0))
            hi = float(np.percentile(img, 99.5))
        else:
            nz = img[img > 0]
            if nz.size == 0:
                return {"vmin": 0.0, "vmax": 1e-9}
            lo = 0.0
            hi = float(np.percentile(nz, 99.5))
        if hi <= lo:
            hi = lo + 1e-9
        return {"vmin": lo, "vmax": hi}

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
