"""Cell navigation and info panel."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)


class _SquareGroupBox(QGroupBox):
    """QGroupBox that constrains itself to a square based on its width."""

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        side = self.width()
        self.setMinimumHeight(side)
        self.setMaximumHeight(side)


# Metrics displayed in the info panel
METRICS = [
    ("SNR (CaImAn)", "snr_caiman"),
    ("SNR2",         "snr2"),
    ("CNN prob.",    "cnn"),
    ("R value",      "rvalue"),
    ("Firing stab.", "firing_stab"),
    ("Peaks avg.",   "peaks_ave"),
    ("Noise (std)",  "noise"),
    ("Skewness",     "skewness"),
]


class NavPanel(QWidget):
    """Cell navigation spinner, category filter, metrics, and component image."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._metric_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._connect_session()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        layout.addWidget(self._build_navigation())
        layout.addWidget(self._build_accept_reject())
        layout.addStretch()
        # Metrics box and component image placed by MainWindow
        self._build_metrics()
        self._build_component_image()

    def _build_navigation(self) -> QGroupBox:
        self.nav_group = QGroupBox("Cell Navigation")
        v = QVBoxLayout(self.nav_group)

        # Spinner row
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell:"))
        self.cell_spinner = QSpinBox()
        self.cell_spinner.setMinimum(0)
        self.cell_spinner.setMaximum(0)
        self.cell_spinner.setFixedWidth(70)
        self.cell_spinner.setToolTip(
            "0-based cell index. Type a number or use arrows to jump to a cell.\n"
            "You can also press ↑/↓ (or ←/→) while this panel has focus to step\n"
            "through the cells matching the current filter (All/Accepted/Rejected)."
        )
        row.addWidget(self.cell_spinner)
        self.total_label = QLabel("/ 0")
        row.addWidget(self.total_label)
        row.addStretch()
        v.addLayout(row)

        # Category filter
        cat_row = QHBoxLayout()
        self._cat_group = QButtonGroup(self)
        cat_tips = {
            "All":      "Step through every cell when navigating with arrow keys.",
            "Accepted": "Restrict arrow-key navigation to currently accepted cells.",
            "Rejected": "Restrict arrow-key navigation to currently rejected cells.",
        }
        for label in ("All", "Accepted", "Rejected"):
            rb = QRadioButton(label)
            rb.setToolTip(cat_tips[label])
            self._cat_group.addButton(rb)
            cat_row.addWidget(rb)
        self._cat_group.buttons()[0].setChecked(True)
        v.addLayout(cat_row)

        return self.nav_group

    def _build_accept_reject(self) -> QGroupBox:
        box = QGroupBox("Manual Edit")
        row = QHBoxLayout(box)
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet("background-color: #2e7d32; color: white;")
        self.accept_btn.setEnabled(False)
        self.accept_btn.setToolTip(
            "Mark the current cell as accepted and flag it as manually overridden,\n"
            "so re-running automatic evaluation will not change its state."
        )
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setStyleSheet("background-color: #c62828; color: white;")
        self.reject_btn.setEnabled(False)
        self.reject_btn.setToolTip(
            "Mark the current cell as rejected and flag it as manually overridden,\n"
            "so re-running automatic evaluation will not change its state."
        )
        row.addWidget(self.accept_btn)
        row.addWidget(self.reject_btn)
        return box

    def _build_metrics(self) -> None:
        """Build the cell metrics widget and store as self.metrics_box."""
        box = QGroupBox("Cell Metrics")
        box.setToolTip(
            "Per-cell metrics for the currently selected cell.\n"
            "Hover individual rows for what each metric measures."
        )
        form = QFormLayout(box)
        form.setSpacing(3)
        tips = {
            "snr_caiman":  "CaImAn SNR — log-probability that the cell shows real activity above noise.",
            "snr2":        "peaks_ave / noise — alternative SNR using mean of top peaks.",
            "cnn":         "CaImAn CNN classifier probability of being a real neuron (0–1).",
            "rvalue":      "Spatial r-value: Pearson correlation between the spatial footprint and the mean pixel activity during detected events.",
            "firing_stab": "Firing stability: number of detected peaks divided by total recording time / window.",
            "peaks_ave":   "Average amplitude of the top-N largest peaks in the trace (after smoothing each ±0.2 s window).",
            "noise":       "Per-cell trace noise standard deviation (Welch PSD in 0.25–0.5 × Nyquist).",
            "skewness":    "Trace skewness — true neurons usually have positively skewed fluorescence.",
        }
        for label, key in METRICS:
            val = QLabel("—")
            val.setAlignment(Qt.AlignRight)
            form.addRow(label + ":", val)
            val.setToolTip(tips.get(key, ""))
            self._metric_labels[key] = val
        self.metrics_box = box

    def _build_component_image(self) -> None:
        """Build the component image widget and store as self.component_box."""
        box = _SquareGroupBox("Component")
        box.setToolTip(
            "Spatial footprint of the currently selected cell, cropped to a square\n"
            "bounding box around its non-zero pixels with a small padding."
        )
        v = QVBoxLayout(box)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(0)
        self.comp_fig = Figure(facecolor="#1e1e1e")
        self.comp_ax = self.comp_fig.add_axes([0, 0, 1, 1])
        self.comp_ax.set_facecolor("#1e1e1e")
        self.comp_ax.axis("off")
        self.comp_canvas = FigureCanvasQTAgg(self.comp_fig)
        self.comp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.comp_canvas)
        self.component_box = box

    # ------------------------------------------------------------------
    # Session wiring
    # ------------------------------------------------------------------

    def _connect_session(self) -> None:
        self.session.add_listener("data_loaded", self._on_data_loaded)
        self.session.add_listener("cell_selected", self.update_cell_info)
        self.session.add_listener("cell_accepted_changed", self._on_accept_changed)
        self.session.add_listener("cells_reevaluated", self.update_counts)

        self.cell_spinner.valueChanged.connect(self._on_spinner_changed)
        self.accept_btn.clicked.connect(
            lambda: self.session.set_accepted(self.session.current_cell, True))
        self.reject_btn.clicked.connect(
            lambda: self.session.set_accepted(self.session.current_cell, False))

    def _on_data_loaded(self) -> None:
        n = self.session.proc.num_cells
        self.cell_spinner.setMaximum(n - 1)
        self.total_label.setText(f"/ {n}")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.update_counts()
        self.update_cell_info(0)

    def _on_spinner_changed(self, value: int) -> None:
        if self.session.est is not None:
            self.session.select_cell(value)

    def _on_accept_changed(self, cell_idx: int) -> None:
        self.update_counts()
        if cell_idx == self.session.current_cell:
            self.update_cell_info(cell_idx)

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_counts(self) -> None:
        """Refresh accepted/rejected count in the navigation group box title."""
        if self.session.proc is None:
            return
        n_acc = int(self.session.proc.accepted.sum())
        n_tot = self.session.proc.num_cells
        self.nav_group.setTitle(
            f"Cell Navigation  —  {n_acc} accepted / {n_tot - n_acc} rejected"
        )

    def update_cell_info(self, cell_idx: int) -> None:
        """Refresh metric labels and component image for the given cell."""
        if self.session.est is None or self.session.proc is None:
            return

        est  = self.session.est
        proc = self.session.proc

        # Keep spinner in sync without re-triggering _on_spinner_changed
        self.cell_spinner.blockSignals(True)
        self.cell_spinner.setValue(cell_idx)
        self.cell_spinner.blockSignals(False)

        # Metric values
        def _get(arr, idx):
            try:
                return float(arr[idx]) if arr is not None else float("nan")
            except (TypeError, IndexError):
                return float("nan")

        vals = {
            "snr_caiman":  _get(est.SNR_comp,              cell_idx),
            "snr2":        _get(proc.SNR2_vals,            cell_idx),
            "cnn":         _get(est.cnn_preds,             cell_idx),
            "rvalue":      _get(est.r_values,              cell_idx),
            "firing_stab": _get(proc.firing_stab_vals,     cell_idx),
            "peaks_ave":   _get(proc.peaks_ave,            cell_idx),
            "noise":       _get(proc.noise,                cell_idx),
            "skewness":    _get(proc.skewness,             cell_idx),
        }
        for key, val in vals.items():
            lbl = self._metric_labels[key]
            lbl.setText("—" if np.isnan(val) else f"{val:.3f}")

        # Highlight current accept/reject state on the buttons
        accepted = bool(proc.accepted[cell_idx])
        self.accept_btn.setStyleSheet(
            "background-color: #1b5e20; color: white; font-weight: bold;"
            if accepted else
            "background-color: #2e7d32; color: white;"
        )
        self.reject_btn.setStyleSheet(
            "background-color: #b71c1c; color: white; font-weight: bold;"
            if not accepted else
            "background-color: #c62828; color: white;"
        )

        # Component spatial footprint image — zoomed to bounding box
        dims = est.dims   # (height, width)
        footprint = est.A[:, cell_idx].toarray().reshape(dims, order="F")

        nz_rows, nz_cols = np.where(footprint > 0)
        if len(nz_rows) > 0:
            r0, r1 = int(nz_rows.min()), int(nz_rows.max())
            c0, c1 = int(nz_cols.min()), int(nz_cols.max())
            half = max((r1 - r0), (c1 - c0)) // 2 + 4  # square half-side + padding
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            r0 = max(0, cr - half);  r1 = min(dims[0] - 1, cr + half)
            c0 = max(0, cc - half);  c1 = min(dims[1] - 1, cc + half)
            crop = footprint[r0:r1+1, c0:c1+1]
        else:
            crop = footprint

        nz = crop[crop > 0]
        vmin = float(np.percentile(nz, 0.5))  if len(nz) else 0
        vmax = float(np.percentile(nz, 99.5)) if len(nz) else 1e-9

        self.comp_ax.cla()
        self.comp_ax.set_position([0, 0, 1, 1])
        self.comp_ax.imshow(crop, cmap="viridis", aspect="equal", origin="lower",
                            interpolation="nearest", vmin=vmin, vmax=vmax)
        self.comp_ax.axis("off")
        self.comp_canvas.draw_idle()

        # Show the current cell number in the group-box title
        state = "accepted" if accepted else "rejected"
        self.component_box.setTitle(f"Component — cell {cell_idx}  ({state})")

    def keyPressEvent(self, event) -> None:
        """Up/down arrows navigate the currently filtered cell list."""
        if self.session.est is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key not in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            super().keyPressEvent(event)
            return

        proc = self.session.proc
        cat_idx = next(
            i for i, btn in enumerate(self._cat_group.buttons()) if btn.isChecked()
        )
        if cat_idx == 1:
            cells = np.where(proc.accepted)[0]
        elif cat_idx == 2:
            cells = np.where(~proc.accepted)[0]
        else:
            cells = np.arange(proc.num_cells)

        if len(cells) == 0:
            return

        cur = self.session.current_cell
        pos = int(np.searchsorted(cells, cur))
        pos = np.clip(pos, 0, len(cells) - 1)

        if key in (Qt.Key_Up, Qt.Key_Left):
            pos = max(pos - 1, 0)
        else:
            pos = min(pos + 1, len(cells) - 1)

        self.cell_spinner.setValue(int(cells[pos]))
