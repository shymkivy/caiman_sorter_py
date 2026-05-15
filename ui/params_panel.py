"""Tabbed parameter panel (Evaluation + Deconvolution)."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class ParamsPanel(QWidget):
    """Tabbed panel: Evaluation params | Deconvolution params."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._build_ui()
        self._connect_session()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.addWidget(self._build_eval_tab())
        # Pre-build deconv panel so run_deconv_btn exists before _connect_session runs.
        self._deconv_panel = self._build_deconv_tab()

    def build_deconv_panel(self) -> QWidget:
        """Return the deconvolution controls widget for placement in the center tab."""
        return self._deconv_panel

    # ---- Evaluation tab -----------------------------------------------

    def _build_eval_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.eval_method_combo = QComboBox()
        self.eval_method_combo.addItems(["CaImAn evaluate", "Reject threshold"])
        self.eval_method_combo.setToolTip(
            "CaImAn evaluate: accept if ANY metric exceeds its upper threshold,\n"
            "then reject if ANY metric falls below its lower threshold.\n\n"
            "Reject threshold: start with all cells accepted, then apply\n"
            "each enabled cutoff in sequence (AND logic)."
        )
        method_row.addWidget(self.eval_method_combo)
        method_row.addStretch()
        v.addLayout(method_row)

        # CaImAn thresholds
        self.caiman_group = QGroupBox("CaImAn thresholds")
        cf = QFormLayout(self.caiman_group)
        self.snr_thresh   = self._spin(0, 100, 2.0)
        self.snr_lowest   = self._spin(0, 100, 0.5)
        self.cnn_thresh   = self._spin(0, 1,   0.99, step=0.01, dec=3)
        self.cnn_lowest   = self._spin(0, 1,   0.10, step=0.01, dec=3)
        self.rval_thresh  = self._spin(-1, 1,  0.80, step=0.05, dec=3)
        self.rval_lowest  = self._spin(-1, 1, -1.00, step=0.05, dec=3)
        self.snr_thresh.setToolTip(
            "Accept if SNR exceeds this value.\n"
            "SNR = -norm.ppf(exp(fitness / N_samples)), where fitness is the\n"
            "log-probability of observing N consecutive exceptional frames."
        )
        self.snr_lowest.setToolTip(
            "Reject if SNR falls below this value, regardless of CNN or r-value.\n"
            "Acts as a hard floor — cells below this are always rejected."
        )
        self.cnn_thresh.setToolTip(
            "Accept if the CNN classifier probability of being a real neuron\n"
            "meets or exceeds this value."
        )
        self.cnn_lowest.setToolTip(
            "Reject if CNN probability falls below this value, regardless of\n"
            "SNR or r-value. Acts as a hard floor."
        )
        self.rval_thresh.setToolTip(
            "Accept if the spatial r-value meets or exceeds this threshold.\n"
            "r-value = Pearson correlation between the component's spatial\n"
            "footprint and the mean pixel activity during detected events."
        )
        self.rval_lowest.setToolTip(
            "Reject if r-value falls below this value, regardless of SNR or CNN.\n"
            "Default -1 means this floor is effectively disabled."
        )
        cf.addRow("SNR thresh:",    self.snr_thresh)
        cf.addRow("SNR lowest:",    self.snr_lowest)
        cf.addRow("CNN thresh:",    self.cnn_thresh)
        cf.addRow("CNN lowest:",    self.cnn_lowest)
        cf.addRow("R-val thresh:",  self.rval_thresh)
        cf.addRow("R-val lowest:",  self.rval_lowest)
        v.addWidget(self.caiman_group)

        # Reject-threshold cutoffs
        _rej_tips = {
            "snr_caiman":      "CaImAn SNR — reject cells whose SNR is below this value.",
            "snr2":            "peaks_ave / noise — reject cells whose peak-based SNR is below this value.",
            "cnn":             "CNN classifier probability — reject cells below this value.",
            "rvalues":         "Spatial r-value — reject cells below this Pearson correlation threshold.",
            "min_sig_frac":    "Minimum active-frame fraction (0–1).\n"
                               "Rejects cells where fewer than this fraction of frames carry signal\n"
                               "(i.e., num_zeros ≥ (1 − min_sig_frac) × n_frames).",
            "firing_stability":"Firing stability — reject cells below this score.\n"
                               "Computed as (number of detected peaks) / (total recording time / window).",
            "skewness":        "Trace skewness — reject cells below this value.\n"
                               "True neurons typically have positively skewed fluorescence traces.",
        }
        self.reject_group = QGroupBox("Reject threshold cutoffs")
        self.reject_group.setEnabled(False)
        rf = QFormLayout(self.reject_group)
        rf.setVerticalSpacing(1)
        rf.setContentsMargins(4, 2, 4, 2)
        self._rej_checks: dict[str, QCheckBox] = {}
        self._rej_spins: dict[str, QDoubleSpinBox] = {}
        for label, key, default in [
            ("SNR (CaImAn)", "snr_caiman",       2.0),
            ("SNR2",         "snr2",              2.0),
            ("CNN",          "cnn",               0.5),
            ("R values",     "rvalues",           0.5),
            ("Min sig frac", "min_sig_frac",      0.1),
            ("Firing stab.", "firing_stability",  0.0),
            ("Skewness",     "skewness",          0.0),
        ]:
            row = QHBoxLayout()
            chk = QCheckBox()
            chk.setToolTip(f"Enable the {label} cutoff")
            spin = self._spin(-1e4, 1e4, default)
            spin.setToolTip(_rej_tips[key])
            row.addWidget(chk)
            row.addWidget(spin)
            rf.addRow(label + ":", self._wrap(row))
            self._rej_checks[key] = chk
            self._rej_spins[key] = spin
        v.addWidget(self.reject_group)

        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate All")
        self.evaluate_btn.setEnabled(False)
        self.evaluate_btn.setToolTip(
            "Re-run automatic evaluation with the current settings.\n"
            "Cells that were manually accepted/rejected are preserved."
        )
        v.addWidget(self.evaluate_btn)
        v.addStretch()

        self.eval_method_combo.currentIndexChanged.connect(self._on_method_changed)
        return w

    # ---- Deconvolution tab --------------------------------------------

    def _build_deconv_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)

        # ---- Spikes (S) post-processing ----------------------------------
        spikes_group = QGroupBox("Spikes (S)")
        sp = QFormLayout(spikes_group)
        self.spikes_smooth       = QCheckBox()
        self.spikes_smooth_sigma = self._spin(1, 10000, 50.0, dec=1)
        self.spikes_scale        = self._spin(0.001, 1e6, 1.0, dec=3)
        self.spikes_shift        = self._spin(-1e6, 1e6, 0.0, dec=3)
        self.spikes_smooth.setToolTip(
            "Apply Gaussian smoothing to the raw CaImAn spike trace (est.S)\n"
            "at display time. Does not alter the stored S array."
        )
        self.spikes_smooth_sigma.setToolTip(
            "Standard deviation of the Gaussian kernel applied to est.S,\n"
            "in milliseconds. Converted to frames via the frame rate."
        )
        self.spikes_scale.setToolTip("Display-only scale applied to the spike trace.")
        self.spikes_shift.setToolTip("Display-only offset applied to the spike trace.")
        sp.addRow("Smooth:",        self.spikes_smooth)
        sp.addRow("Smooth σ (ms):", self.spikes_smooth_sigma)
        sp.addRow("Scale:",         self.spikes_scale)
        sp.addRow("Shift:",         self.spikes_shift)
        v.addWidget(spikes_group)

        # ---- Smooth dF/dt -------------------------------------------------
        dfdt_group = QGroupBox("Smooth dF/dt")
        df = QFormLayout(dfdt_group)
        self.dfdt_sigma     = self._spin(1, 10000, 50.0, dec=1)
        self.dfdt_rectify   = QCheckBox()
        self.dfdt_normalize = QCheckBox()
        self.dfdt_thresh    = QCheckBox()
        self.dfdt_thresh_z  = self._spin(0, 100, 1.0, dec=2)
        self.dfdt_scale     = self._spin(0.001, 1e6, 1.0, dec=3)
        self.dfdt_shift     = self._spin(-1e6, 1e6, 0.0, dec=3)
        self.dfdt_sigma.setToolTip(
            "Standard deviation of the Gaussian kernel used to smooth diff(C + YrA).\n"
            "Specified in milliseconds — converted to frames using the frame rate."
        )
        self.dfdt_rectify.setToolTip("Set negative values to zero after smoothing (half-wave rectification).")
        self.dfdt_normalize.setToolTip("Divide each cell's output by its absolute peak so traces are bounded by ±1.")
        self.dfdt_thresh.setToolTip("Zero out values below threshold_z × std (per-cell).")
        self.dfdt_thresh_z.setToolTip("Z-score cutoff applied when 'Apply thresh' is checked.")
        self.dfdt_scale.setToolTip("Display-only scale applied to the final output trace.")
        self.dfdt_shift.setToolTip("Display-only offset applied to the final output trace.")
        df.addRow("Gauss σ (ms):", self.dfdt_sigma)
        df.addRow("Rectify:",      self.dfdt_rectify)
        df.addRow("Normalize:",    self.dfdt_normalize)
        df.addRow("Apply thresh:", self.dfdt_thresh)
        df.addRow("Thresh (z):",   self.dfdt_thresh_z)
        df.addRow("Scale:",        self.dfdt_scale)
        df.addRow("Shift:",        self.dfdt_shift)
        v.addWidget(dfdt_group)

        # ---- Constrained foopsi ------------------------------------------
        foopsi_group = QGroupBox("Constrained foopsi")
        ff = QFormLayout(foopsi_group)
        self.foopsi_solver  = QComboBox()
        # Label unavailable solvers so user knows up-front what works
        from caiman_sorter_py.core.deconvolution import solver_available
        labels = []
        for key, name in [("oasis", "OASIS"), ("cvxpy", "CVXPY"), ("cvx", "CVX")]:
            labels.append(name if solver_available(key) else f"{name} (not installed)")
        self.foopsi_solver.addItems(labels)
        self.foopsi_solver.setToolTip(
            "OASIS:  fast (~0.1–1 s per cell), recommended default.\n"
            "CVXPY:  interior-point convex solver, much slower (~10–60 s per cell).\n"
            "        Requires `pip install cvxpy`.\n"
            "CVX:    older cvxopt+picos solver, slowest. Requires `pip install cvxopt picos`."
        )
        self.foopsi_ar      = QComboBox(); self.foopsi_ar.addItems(["AR1", "AR2"])
        self.foopsi_manual  = QCheckBox()
        self.foopsi_tau_d   = self._spin(0.001, 100, 0.4, dec=3)
        self.foopsi_tau_r   = self._spin(0.001, 100, 0.1, dec=3)
        self.foopsi_fudge    = self._spin(0.5, 1.0, 0.99, step=0.005, dec=3)
        self.foopsi_smooth_s = QCheckBox()
        self.foopsi_smooth_sigma = self._spin(1, 10000, 50.0, dec=1)
        self.foopsi_scale    = self._spin(0.001, 1e6, 1.0, dec=3)
        self.foopsi_shift    = self._spin(-1e6, 1e6, 0.0, dec=3)
        self.foopsi_ar.setToolTip(
            "AR(1): single exponential decay (one tau).\n"
            "AR(2): separate rise and decay — better for GCaMP6s and similar."
        )
        self.foopsi_manual.setToolTip(
            "If checked, convert the manual τ values below to AR coefficients.\n"
            "If unchecked, use the per-cell gAR1/gAR2 estimated during proc init."
        )
        self.foopsi_tau_d.setToolTip("Decay τ in seconds (only used when 'Manual τ' is checked).")
        self.foopsi_tau_r.setToolTip("Rise τ in seconds, AR(2) only (only used when 'Manual τ' is checked).")
        self.foopsi_fudge.setToolTip(
            "Fudge factor applied to AR coefficients to reduce time-constant bias.\n"
            "Each AR root is multiplied by this value (shrinks toward 0 → faster decay).\n"
            "Default 0.99 — values near 1 keep the estimate closer to raw."
        )
        self.foopsi_smooth_s.setToolTip(
            "Apply Gaussian smoothing to the foopsi spike trace at display time.\n"
            "Does not require re-running foopsi — updates the plot live."
        )
        self.foopsi_smooth_sigma.setToolTip(
            "Standard deviation of the Gaussian kernel applied to the foopsi spike trace,\n"
            "in milliseconds. Converted to frames via the frame rate."
        )
        self.foopsi_scale.setToolTip("Display-only scale applied to the deconvolved spike train.")
        self.foopsi_shift.setToolTip("Display-only offset applied to the deconvolved spike train.")
        ff.addRow("Solver:",         self.foopsi_solver)
        ff.addRow("AR order:",       self.foopsi_ar)
        ff.addRow("Manual τ:",       self.foopsi_manual)
        ff.addRow("τ decay (s):",    self.foopsi_tau_d)
        ff.addRow("τ rise (s):",     self.foopsi_tau_r)
        ff.addRow("Fudge factor:",   self.foopsi_fudge)
        ff.addRow("Smooth S:",       self.foopsi_smooth_s)
        ff.addRow("Smooth σ (ms):",  self.foopsi_smooth_sigma)
        ff.addRow("Scale:",          self.foopsi_scale)
        ff.addRow("Shift:",          self.foopsi_shift)
        self.run_foopsi_btn = QPushButton("Run foopsi")
        self.run_foopsi_btn.setEnabled(False)
        self.run_foopsi_btn.setToolTip(
            "Run constrained foopsi / OASIS on every cell (or only the current\n"
            "cell if 'Current cell only' is checked below).\n"
            "  • Single cell: runs synchronously, updates the trace immediately.\n"
            "  • All cells: spawns a background worker with a progress dialog;\n"
            "    parallelised across cells via joblib threads (~0.5–1 s/cell with OASIS).\n"
            "Fails fast with a friendly install hint if the selected solver isn't installed."
        )
        ff.addRow(self.run_foopsi_btn)
        v.addWidget(foopsi_group)

        # ---- Shared options ----------------------------------------------
        self.current_cell_only_chk = QCheckBox("Current cell only")
        self.current_cell_only_chk.setToolTip(
            "If checked, run the selected method for the currently selected cell only.\n"
            "Useful for quick parameter tuning."
        )
        v.addWidget(self.current_cell_only_chk)

        # MCMC dropped — modern CaImAn Python no longer ships cont_ca_sampler.
        v.addStretch()

        # Live trace refresh for display-only knobs (scale/shift/smoothing)
        for spin in (self.dfdt_scale, self.dfdt_shift,
                     self.dfdt_sigma, self.dfdt_thresh_z,
                     self.foopsi_scale, self.foopsi_shift,
                     self.foopsi_smooth_sigma,
                     self.spikes_smooth_sigma,
                     self.spikes_scale, self.spikes_shift):
            spin.valueChanged.connect(self._on_deconv_display_change)
        for chk in (self.dfdt_rectify, self.dfdt_normalize, self.dfdt_thresh,
                    self.foopsi_smooth_s, self.spikes_smooth):
            chk.toggled.connect(self._on_deconv_display_change)

        # Manual-tau toggles enable state of tau spinboxes
        self.foopsi_manual.toggled.connect(self._on_foopsi_manual_toggled)
        self.foopsi_ar.currentIndexChanged.connect(self._on_foopsi_ar_changed)
        self._on_foopsi_manual_toggled(self.foopsi_manual.isChecked())
        self._on_foopsi_ar_changed(self.foopsi_ar.currentIndex())

        return w

    def _on_foopsi_manual_toggled(self, on: bool) -> None:
        """Enable τ spinboxes only when 'Manual τ' is checked."""
        self.foopsi_tau_d.setEnabled(on)
        self.foopsi_tau_r.setEnabled(on and self.foopsi_ar.currentIndex() == 1)

    def _on_foopsi_ar_changed(self, idx: int) -> None:
        """Grey out τ rise when AR(1) is selected (it's not used)."""
        is_ar2 = idx == 1
        self.foopsi_tau_r.setEnabled(is_ar2 and self.foopsi_manual.isChecked())

    # ------------------------------------------------------------------
    # Session wiring
    # ------------------------------------------------------------------

    def _connect_session(self) -> None:
        self.session.add_listener("data_loaded", self._on_data_loaded)
        self.evaluate_btn.clicked.connect(self._on_evaluate)
        self.run_foopsi_btn.clicked.connect(self._on_run_foopsi)

    def _on_data_loaded(self) -> None:
        self.evaluate_btn.setEnabled(True)
        self.run_foopsi_btn.setEnabled(True)
        # Seed CaImAn thresholds from the values stored in the file
        ep = self.session.est.eval_params_caiman or {}
        ec = self.session.ops.eval_caiman
        ec.snr_thresh        = ep.get("snr_thresh",        ec.snr_thresh)
        ec.snr_lowest_thresh = ep.get("snr_lowest_thresh", ec.snr_lowest_thresh)
        ec.cnn_thresh        = ep.get("cnn_thresh",        ec.cnn_thresh)
        ec.cnn_lowest_thresh = ep.get("cnn_lowest_thresh", ec.cnn_lowest_thresh)
        ec.rval_thresh       = ep.get("rval_thresh",       ec.rval_thresh)
        ec.rval_lowest_thresh = ep.get("rval_lowest_thresh", ec.rval_lowest_thresh)
        self.load_ops()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_ops(self) -> None:
        """Populate all controls from session.ops."""
        ops = self.session.ops
        ec  = ops.eval_caiman
        er  = ops.eval_reject

        # Method selector
        self.eval_method_combo.setCurrentIndex(0 if ops.eval_method == "caiman" else 1)

        # CaImAn thresholds
        self.snr_thresh.setValue(ec.snr_thresh)
        self.snr_lowest.setValue(ec.snr_lowest_thresh)
        self.cnn_thresh.setValue(ec.cnn_thresh)
        self.cnn_lowest.setValue(ec.cnn_lowest_thresh)
        self.rval_thresh.setValue(ec.rval_thresh)
        self.rval_lowest.setValue(ec.rval_lowest_thresh)

        # Reject-threshold checks + values
        for key, use_attr, val_attr in [
            ("snr_caiman",      "use_snr_caiman",      "snr_caiman"),
            ("snr2",            "use_snr2",            "snr2"),
            ("cnn",             "use_cnn",             "cnn"),
            ("rvalues",         "use_rvalues",         "rvalues"),
            ("min_sig_frac",    "use_min_sig_frac",    "min_sig_frac"),
            ("firing_stability","use_firing_stability","firing_stability"),
            ("skewness",        "use_skewness",        "skewness"),
        ]:
            self._rej_checks[key].setChecked(getattr(er, use_attr))
            self._rej_spins[key].setValue(getattr(er, val_attr))

        # Deconv params
        sp = ops.spikes
        self.spikes_smooth.setChecked(sp.smooth)
        self.spikes_smooth_sigma.setValue(sp.smooth_sigma)
        self.spikes_scale.setValue(sp.scale)
        self.spikes_shift.setValue(sp.shift)

        sd = ops.smooth_dfdt
        self.dfdt_sigma.setValue(sd.gauss_sigma)
        self.dfdt_rectify.setChecked(sd.rectify)
        self.dfdt_normalize.setChecked(sd.normalize)
        self.dfdt_thresh.setChecked(sd.apply_thresh)
        self.dfdt_thresh_z.setValue(sd.threshold_z)
        self.dfdt_scale.setValue(sd.scale)
        self.dfdt_shift.setValue(sd.shift)

        fp = ops.foopsi
        solver_idx = {"oasis": 0, "cvxpy": 1, "cvx": 2}.get(fp.solver, 0)
        self.foopsi_solver.setCurrentIndex(solver_idx)
        self.foopsi_ar.setCurrentIndex(fp.ar_order - 1)
        self.foopsi_manual.setChecked(fp.manual_tau)
        self.foopsi_tau_d.setValue(fp.tau_decay)
        self.foopsi_tau_r.setValue(fp.tau_rise)
        self.foopsi_fudge.setValue(fp.fudge_factor)
        self.foopsi_smooth_s.setChecked(fp.smooth_s)
        self.foopsi_smooth_sigma.setValue(fp.smooth_sigma)
        self.foopsi_scale.setValue(fp.scale)
        self.foopsi_shift.setValue(fp.shift)

    def sync_to_ops(self) -> None:
        """Write all current control values into session.ops.

        NOTE: When adding new Ops / sub-param fields, add the corresponding
        read here (and the counterpart write in load_ops) so the values are
        persisted across sessions.
        """
        ops = self.session.ops
        ec  = ops.eval_caiman
        er  = ops.eval_reject

        ops.eval_method = "caiman" if self.eval_method_combo.currentIndex() == 0 \
                          else "reject_threshold"

        # CaImAn thresholds
        ec.snr_thresh         = self.snr_thresh.value()
        ec.snr_lowest_thresh  = self.snr_lowest.value()
        ec.cnn_thresh         = self.cnn_thresh.value()
        ec.cnn_lowest_thresh  = self.cnn_lowest.value()
        ec.rval_thresh        = self.rval_thresh.value()
        ec.rval_lowest_thresh = self.rval_lowest.value()

        # Reject-threshold checks + values
        for key, use_attr, val_attr in [
            ("snr_caiman",       "use_snr_caiman",      "snr_caiman"),
            ("snr2",             "use_snr2",            "snr2"),
            ("cnn",              "use_cnn",             "cnn"),
            ("rvalues",          "use_rvalues",         "rvalues"),
            ("min_sig_frac",     "use_min_sig_frac",    "min_sig_frac"),
            ("firing_stability", "use_firing_stability","firing_stability"),
            ("skewness",         "use_skewness",        "skewness"),
        ]:
            setattr(er, use_attr, self._rej_checks[key].isChecked())
            setattr(er, val_attr, self._rej_spins[key].value())

        # Spikes (display-only smoothing/scale/shift of est.S)
        sp = ops.spikes
        sp.smooth       = self.spikes_smooth.isChecked()
        sp.smooth_sigma = self.spikes_smooth_sigma.value()
        sp.scale        = self.spikes_scale.value()
        sp.shift        = self.spikes_shift.value()

        # Smooth dF/dt
        sd = ops.smooth_dfdt
        sd.gauss_sigma  = self.dfdt_sigma.value()
        sd.rectify      = self.dfdt_rectify.isChecked()
        sd.normalize    = self.dfdt_normalize.isChecked()
        sd.apply_thresh = self.dfdt_thresh.isChecked()
        sd.threshold_z  = self.dfdt_thresh_z.value()
        sd.scale        = self.dfdt_scale.value()
        sd.shift        = self.dfdt_shift.value()

        # Foopsi
        fp = ops.foopsi
        fp.solver       = ["oasis", "cvxpy", "cvx"][self.foopsi_solver.currentIndex()]
        fp.ar_order     = self.foopsi_ar.currentIndex() + 1
        fp.manual_tau   = self.foopsi_manual.isChecked()
        fp.tau_decay    = self.foopsi_tau_d.value()
        fp.tau_rise     = self.foopsi_tau_r.value()
        fp.fudge_factor = self.foopsi_fudge.value()
        fp.smooth_s     = self.foopsi_smooth_s.isChecked()
        fp.smooth_sigma = self.foopsi_smooth_sigma.value()
        fp.scale        = self.foopsi_scale.value()
        fp.shift        = self.foopsi_shift.value()

    def _on_evaluate(self) -> None:
        """Sync controls → ops, then re-evaluate."""
        from caiman_sorter_py.core.evaluation import evaluate_components, update_accepted
        self.sync_to_ops()
        ops = self.session.ops
        core_mask = evaluate_components(self.session.est, self.session.proc, ops)
        update_accepted(self.session.proc, core_mask)
        self.session.reevaluate_all()

    def _cells_to_process(self) -> "np.ndarray":
        """Return cell indices to deconvolve: current cell only, or all."""
        import numpy as np
        if self.current_cell_only_chk.isChecked():
            return np.array([int(self.session.current_cell)])
        return np.arange(self.session.proc.num_cells)

    def _on_run_foopsi(self) -> None:
        """Run constrained foopsi for current cell or all cells.

        Always routed through MainWindow.run_foopsi_async so the user gets the
        modal "Running Constrained foopsi" dialog regardless of cell count —
        even a single cell can take ~0.5–1 s and a silent UI freeze feels
        broken. Warns the user if the selected solver isn't installed.
        """
        from caiman_sorter_py.core.deconvolution import (
            run_foopsi, solver_available, SOLVER_INSTALL_HINT,
        )
        from PyQt5.QtWidgets import QMessageBox
        self.sync_to_ops()
        solver = self.session.ops.foopsi.solver
        if not solver_available(solver):
            QMessageBox.warning(
                self, "Solver not installed",
                f"The '{solver}' solver is not available in this environment.\n\n"
                f"Install it with:\n    {SOLVER_INSTALL_HINT.get(solver, '?')}\n\n"
                f"Or switch the Solver dropdown to OASIS (always available)."
            )
            return
        cells = self._cells_to_process()

        main = self.window()
        if hasattr(main, "run_foopsi_async"):
            main.run_foopsi_async(cells)
        else:
            # Headless / detached fallback — no parent window to host the dialog
            try:
                run_foopsi(self.session.est, self.session.proc, self.session.ops,
                           cells=cells)
            except Exception as exc:
                import traceback
                QMessageBox.critical(self, "foopsi error",
                                     f"{exc}\n\n{traceback.format_exc()}")
                return
            self.session.refresh_cell()

    def _on_deconv_display_change(self) -> None:
        """Write display-only deconv knobs to ops and refresh the trace.

        Covers everything that affects the rendered traces without needing
        a deconvolution re-run:
          - spikes smoothing / scale / shift
          - smooth dF/dt sigma, rectify, normalize, threshold, scale, shift
            (the trace recomputes live in TracePanel from these params)
          - foopsi (S) smoothing, scale, shift
        """
        ops = self.session.ops
        # Spikes (S)
        ops.spikes.smooth         = self.spikes_smooth.isChecked()
        ops.spikes.smooth_sigma   = self.spikes_smooth_sigma.value()
        ops.spikes.scale          = self.spikes_scale.value()
        ops.spikes.shift          = self.spikes_shift.value()
        # Smooth dF/dt (display recomputes from these)
        ops.smooth_dfdt.gauss_sigma  = self.dfdt_sigma.value()
        ops.smooth_dfdt.rectify      = self.dfdt_rectify.isChecked()
        ops.smooth_dfdt.normalize    = self.dfdt_normalize.isChecked()
        ops.smooth_dfdt.apply_thresh = self.dfdt_thresh.isChecked()
        ops.smooth_dfdt.threshold_z  = self.dfdt_thresh_z.value()
        ops.smooth_dfdt.scale        = self.dfdt_scale.value()
        ops.smooth_dfdt.shift        = self.dfdt_shift.value()
        # Foopsi (display-only knobs)
        ops.foopsi.scale          = self.foopsi_scale.value()
        ops.foopsi.shift          = self.foopsi_shift.value()
        ops.foopsi.smooth_s       = self.foopsi_smooth_s.isChecked()
        ops.foopsi.smooth_sigma   = self.foopsi_smooth_sigma.value()
        if self.session.est is not None:
            self.session.refresh_cell()

    def _on_method_changed(self, idx: int) -> None:
        is_caiman = idx == 0
        self.caiman_group.setEnabled(is_caiman)
        self.reject_group.setEnabled(not is_caiman)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spin(lo, hi, val, step=0.1, dec=2) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setFixedWidth(80)
        return s

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        return w
