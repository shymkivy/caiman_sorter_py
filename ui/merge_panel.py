"""Duplicate-component detection / merge panel."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from matplotlib.figure import Figure

from caiman_sorter_py.core.merge import METHODS


class _MinimalToolbar(NavigationToolbar2QT):
    """matplotlib toolbar trimmed to Home / Pan / Zoom (same as TracePanel)."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ("Home", "Pan", "Zoom")]


class _MergePreviewDialog(QDialog):
    """Non-modal window showing two cells + their weighted-average merge preview.

    Toolbar (Home / Pan / Zoom) lets you box-zoom on the trace axis just like
    the trace panel in the main window.
    """

    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        a, b = payload["cell_a"], payload["cell_b"]
        self.setWindowTitle(f"Merge preview — cells {a} and {b}")
        self.setMinimumSize(900, 640)
        self.setModal(False)              # let user interact with main window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._fig    = Figure(facecolor="#1e1e1e", tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        toolbar = _MinimalToolbar(self._canvas, self)
        toolbar.setMaximumHeight(28)
        toolbar.setToolTip(
            "Home: reset view. Pan: drag to move. Zoom: drag a rectangle to box-zoom."
        )

        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, stretch=1)

        self._draw(self._fig, payload)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw(fig, p: dict) -> None:
        # Make the trace much taller than the footprints — it's the focus.
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.2],
                              hspace=0.25, wspace=0.15)

        a, b = p["cell_a"], p["cell_b"]
        r0, r1, c0, c1 = p["crop"]

        def _imshow(ax, img, title):
            crop = img[r0:r1 + 1, c0:c1 + 1]
            nz = crop[crop > 0]
            vmin = float(np.percentile(nz,  0.5)) if nz.size else 0.0
            vmax = float(np.percentile(nz, 99.5)) if nz.size else 1e-9
            ax.imshow(crop, cmap="viridis", origin="lower", aspect="equal",
                      interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")

        ax_a = fig.add_subplot(gs[0, 0]);  _imshow(ax_a, p["A1_2d"], f"A — cell {a}")
        ax_b = fig.add_subplot(gs[0, 1]);  _imshow(ax_b, p["A2_2d"], f"A — cell {b}")
        ax_m = fig.add_subplot(gs[0, 2]);  _imshow(ax_m, p["A_merged_2d"], "A — merged (weighted ave)")

        ax_tr = fig.add_subplot(gs[1, :])
        fr = p["fr"]
        t = np.arange(p["trace1"].size) / fr
        ax_tr.plot(t, p["trace_merged"], color="#bbbbbb", lw=1.4,
                   label="merged", alpha=0.9, zorder=3)
        ax_tr.plot(t, p["trace1"], color="#d95f02", lw=0.9, label=f"cell {a}")
        ax_tr.plot(t, p["trace2"], color="#1f78b4", lw=0.9, label=f"cell {b}")
        ax_tr.set_facecolor("#1e1e1e")
        ax_tr.tick_params(colors="gray")
        for spine in ("bottom", "left"):
            ax_tr.spines[spine].set_color("gray")
        for spine in ("top", "right"):
            ax_tr.spines[spine].set_visible(False)
        ax_tr.set_xlabel("Time (s)", color="gray")
        ax_tr.set_ylabel("Fluorescence", color="gray")
        ax_tr.set_xlim(0, t[-1] if t.size else 1)
        ax_tr.set_title(
            f"spatial overlap = {p['spatial_overlap']:.3f}, "
            f"temporal corr = {p['temporal_corr']:.3f}",
            color="white", fontsize=10,
        )
        leg = ax_tr.legend(loc="upper right", facecolor="#1e1e1e",
                            labelcolor="white", framealpha=0.8)
        for txt in leg.get_texts():
            txt.set_color("white")


class MergePanel(QWidget):
    """Find pairs of components likely to be the same neuron, then optionally act on them."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._last_pairs: list = []
        self._build_ui()
        self._connect_session()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_results_group(), stretch=1)
        layout.addWidget(self._build_actions_row())

    def _build_params_group(self) -> QGroupBox:
        box = QGroupBox("Duplicate-component detection")
        f = QFormLayout(box)

        self.method_combo = QComboBox()
        for m in METHODS:
            self.method_combo.addItem(m)
        self.method_combo.setToolTip(
            "How to act on identified duplicate pairs:\n"
            "  • choose best snr — reject the lower-SNR2 cell of each pair (recommended,\n"
            "    reversible, no new components added).\n"
            "  • weighted ave / full mean corr / svd / nmf — create a new merged component\n"
            "    (not yet implemented in the Python port; ported from MATLAB f_cs_find_similar_comp_core)."
        )

        self.spatial_thr = QDoubleSpinBox()
        self.spatial_thr.setRange(0.0, 1e6)
        self.spatial_thr.setSingleStep(0.05)
        self.spatial_thr.setDecimals(3)
        self.spatial_thr.setValue(self.session.ops.merge.spatial_thr)
        self.spatial_thr.setToolTip(
            "Spatial overlap threshold. Computed as the (i, j) entry of A^T A —\n"
            "i.e. the inner product of the two cells' spatial footprints.\n"
            "Larger = stronger pixel overlap required. Typical: 0.3–1.0."
        )

        self.temporal_thr = QDoubleSpinBox()
        self.temporal_thr.setRange(-1.0, 1.0)
        self.temporal_thr.setSingleStep(0.05)
        self.temporal_thr.setDecimals(3)
        self.temporal_thr.setValue(self.session.ops.merge.temporal_thr)
        self.temporal_thr.setToolTip(
            "Pearson correlation threshold between (C + YrA) traces of the two cells.\n"
            "Pairs above both thresholds are flagged as the same neuron."
        )

        self.use_acc_chk = QCheckBox("Compare only accepted cells")
        self.use_acc_chk.setChecked(self.session.ops.merge.use_accepted_only)
        self.use_acc_chk.setToolTip(
            "If checked, only currently accepted cells are considered for duplicate detection.\n"
            "Avoids re-flagging already-rejected components."
        )

        f.addRow("Method:",            self.method_combo)
        f.addRow("Spatial thresh:",    self.spatial_thr)
        f.addRow("Temporal thresh:",   self.temporal_thr)
        f.addRow(self.use_acc_chk)
        return box

    def _build_results_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        v = QVBoxLayout(box)
        v.setSpacing(4)

        self.summary_lbl = QLabel("Click 'Find duplicates' to scan.")
        v.addWidget(self.summary_lbl)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Cell A", "Cell B", "Spatial", "Temporal", "Result"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setToolTip(
            "Flagged duplicate pairs, sorted by spatial overlap descending.\n"
            "Click a row to jump to Cell A in the navigation panel."
        )
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.itemDoubleClicked.connect(lambda *_: self._on_plot())
        v.addWidget(self.table, stretch=1)
        return box

    def _build_actions_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)

        self.find_btn = QPushButton("Find duplicates")
        self.find_btn.setEnabled(False)
        self.find_btn.setToolTip("Scan the current session for duplicate-component pairs.")
        self.find_btn.clicked.connect(self._on_find)

        self.plot_btn = QPushButton("Plot pair")
        self.plot_btn.setEnabled(False)
        self.plot_btn.setToolTip(
            "Open a window with the two cells' spatial footprints, the\n"
            "weighted-average merge preview, and all three traces overlaid.\n"
            "Tip: double-click a row in the table for the same effect."
        )
        self.plot_btn.clicked.connect(self._on_plot)

        self.apply_btn = QPushButton("Apply merge")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip(
            "Act on the pairs above using the selected method.\n"
            "For 'choose best snr' this rejects the lower-SNR2 cell of each pair\n"
            "and flags it as manually overridden."
        )
        self.apply_btn.clicked.connect(self._on_apply)

        row.addWidget(self.find_btn)
        row.addWidget(self.plot_btn)
        row.addWidget(self.apply_btn)
        row.addStretch()
        return w

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _connect_session(self) -> None:
        self.session.add_listener("data_loaded", self._on_data_loaded)
        # Param-control changes write straight to ops
        self.method_combo.currentTextChanged.connect(
            lambda t: setattr(self.session.ops.merge, "method", t)
        )
        self.spatial_thr.valueChanged.connect(
            lambda v: setattr(self.session.ops.merge, "spatial_thr", v)
        )
        self.temporal_thr.valueChanged.connect(
            lambda v: setattr(self.session.ops.merge, "temporal_thr", v)
        )
        self.use_acc_chk.toggled.connect(
            lambda b: setattr(self.session.ops.merge, "use_accepted_only", b)
        )

    def _on_data_loaded(self) -> None:
        self.find_btn.setEnabled(True)
        self.apply_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.summary_lbl.setText("Click 'Find duplicates' to scan.")
        self._last_pairs = []
        self.load_ops()

    def load_ops(self) -> None:
        """Sync widgets from session.ops.merge."""
        mp = self.session.ops.merge
        idx = METHODS.index(mp.method) if mp.method in METHODS else 0
        self.method_combo.setCurrentIndex(idx)
        self.spatial_thr.setValue(mp.spatial_thr)
        self.temporal_thr.setValue(mp.temporal_thr)
        self.use_acc_chk.setChecked(mp.use_accepted_only)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_find(self) -> None:
        from caiman_sorter_py.core.merge import find_duplicate_pairs
        if self.session.est is None:
            return
        try:
            pairs = find_duplicate_pairs(
                self.session.est, self.session.proc, self.session.ops,
            )
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Find error",
                                 f"{exc}\n\n{traceback.format_exc()}")
            return

        self._last_pairs = pairs
        self.apply_btn.setEnabled(len(pairs) > 0)
        self._fill_table(pairs)
        self.summary_lbl.setText(
            f"{len(pairs)} duplicate pair{'s' if len(pairs) != 1 else ''} found."
        )

    def _on_apply(self) -> None:
        from caiman_sorter_py.core.merge import apply_merge
        if not self._last_pairs:
            return
        try:
            result = apply_merge(
                self.session.est, self.session.proc, self.session.ops, self._last_pairs,
            )
        except NotImplementedError as exc:
            QMessageBox.warning(self, "Method not implemented", str(exc))
            return
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Apply error",
                                 f"{exc}\n\n{traceback.format_exc()}")
            return

        n_changed = int(result.get("n_changed", 0))
        self.summary_lbl.setText(
            f"Applied '{self.session.ops.merge.method}' — {n_changed} cell(s) modified."
        )
        # Annotate the Result column with the kept cell per pair
        self._fill_result_column(result)
        # Re-emit so image / nav / metrics panels refresh
        self.session.reevaluate_all()
        # New duplicate scan should now find fewer hits
        self.apply_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_table(self, pairs) -> None:
        self.table.setRowCount(len(pairs))
        for r, p in enumerate(pairs):
            for c, val in enumerate([p.cell_a, p.cell_b,
                                     f"{p.spatial_overlap:.4f}",
                                     f"{p.temporal_corr:.4f}",
                                     ""]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

    def _fill_result_column(self, result: dict) -> None:
        """Populate the 'Result' column after a merge has been applied.

        For 'choose best snr', shows the kept cell ID and notes the rejected one
        as a tooltip. Robust to result dicts that may be missing keys.
        """
        kept = result.get("kept", [])
        rejected = result.get("rejected", [])
        for r in range(self.table.rowCount()):
            kept_id = kept[r] if r < len(kept) else None
            rej_id  = rejected[r] if r < len(rejected) else None
            text = "—" if kept_id is None else f"kept {kept_id}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if rej_id is not None:
                item.setToolTip(f"Kept cell {kept_id}, rejected cell {rej_id}.")
            self.table.setItem(r, 4, item)

    def _on_row_selected(self) -> None:
        """Jump to Cell A of the selected pair; enable Plot button."""
        rows = self.table.selectionModel().selectedRows()
        has_sel = bool(rows) and bool(self._last_pairs)
        self.plot_btn.setEnabled(has_sel)
        if not has_sel:
            return
        row = rows[0].row()
        if 0 <= row < len(self._last_pairs):
            self.session.select_cell(self._last_pairs[row].cell_a)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _current_pair(self):
        """Return the DuplicatePair for the currently selected row, or None."""
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self._last_pairs:
            return None
        r = rows[0].row()
        return self._last_pairs[r] if 0 <= r < len(self._last_pairs) else None

    def _on_plot(self) -> None:
        pair = self._current_pair()
        if pair is None or self.session.est is None:
            return
        from caiman_sorter_py.core.merge import weighted_ave_preview
        try:
            payload = weighted_ave_preview(self.session.est, pair)
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Plot error",
                                 f"{exc}\n\n{traceback.format_exc()}")
            return

        # Keep a reference so the dialog isn't garbage-collected
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs = []
        dlg = _MergePreviewDialog(payload, self.window())
        # Clean up reference when the dialog closes
        dlg.finished.connect(lambda _: self._open_dialogs.remove(dlg)
                              if dlg in self._open_dialogs else None)
        self._open_dialogs.append(dlg)
        dlg.show()
