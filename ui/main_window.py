"""Main application window."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QSplitter, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox,
)

from dataclasses import fields

from caiman_sorter_py import __version__
from caiman_sorter_py.core.state import (
    OPS_SUB_PREFIXES, OPS_TOP_FIELD_NAMES, QSETTINGS_ONLY_SUBS, Session,
)
from caiman_sorter_py.ui.image_panel import ImagePanel
from caiman_sorter_py.ui.merge_panel import MergePanel
from caiman_sorter_py.ui.nav_panel import NavPanel
from caiman_sorter_py.ui.params_panel import ParamsPanel
from caiman_sorter_py.ui.trace_panel import TracePanel

_SETTINGS_ORG  = "CaimanSorter"
_SETTINGS_APP  = "CaimanSorterPy"


class _LoadWorker(QThread):
    """Runs HDF5 load + proc init off the main thread.

    Handles both a fresh CaImAn HDF5 (runs initialize_proc) and a saved
    session file (restores est/proc/ops directly, no proc init needed).

    Produces brand-new `est` and `proc` objects — nothing in the live Session
    is mutated until `MainWindow._on_load_succeeded` calls `session.load_data`.
    `self.ops` is read-only here (used to gate behaviour like load_caiman_rejected),
    so concurrent reads/writes are safe.
    """
    progress  = pyqtSignal(str)
    succeeded = pyqtSignal(object, object, object)   # est, proc, ops_or_None
    failed    = pyqtSignal(str, str)                 # message, traceback

    def __init__(self, path: str, ops):
        super().__init__()
        self.path = path
        self.ops  = ops

    def run(self) -> None:
        try:
            from caiman_sorter_py.io.session import detect_format
            fmt = detect_format(self.path)

            if fmt == "session":
                from caiman_sorter_py.io.session import load_session
                self.progress.emit("Restoring saved session…")
                est, proc, ops = load_session(self.path)
                self.progress.emit(
                    f"Session restored: {proc.num_cells} cells, "
                    f"{proc.num_frames} frames."
                )
                self.succeeded.emit(est, proc, ops)
                return

            if fmt == "session_mat":
                from caiman_sorter_py.io.mat_loader import load_session_mat
                self.progress.emit("Loading MATLAB sort .mat…")
                est, proc, ops = load_session_mat(self.path)
                self.progress.emit(
                    f"Session restored: {proc.num_cells} cells, "
                    f"{proc.num_frames} frames."
                )
                self.succeeded.emit(est, proc, ops)
                return

            # Default: treat as a CaImAn HDF5
            from caiman_sorter_py.io.hdf5_loader import load_hdf5
            from caiman_sorter_py.core.proc_init import initialize_proc
            est = load_hdf5(self.path,
                            load_rejected=self.ops.load_caiman_rejected,
                            contour_thr=self.ops.contour_thr)
            self.progress.emit(
                f"HDF5 loaded: {est.A.shape[1]} cells, {est.C.shape[1]} frames."
            )
            proc = initialize_proc(est, self.ops,
                                   log_cb=lambda m: self.progress.emit(m))
            self.succeeded.emit(est, proc, None)
        except Exception as exc:
            import traceback
            self.failed.emit(str(exc), traceback.format_exc())


class _FoopsiWorker(QThread):
    """Runs constrained foopsi for many cells off the main thread.

    Writes per-cell results into `proc.foopsi.{S,C,g}[idx]` in place. This is
    safe **only because** the worker is always launched alongside a
    `Qt.ApplicationModal` `_FoopsiDialog` (`setModal(True) + exec_()`), which
    blocks all input to the rest of the application — so the main thread can't
    mutate or read `proc.foopsi` while this worker runs. CPython's GIL also
    guarantees that individual list-element writes are atomic.

    If you ever launch this worker without a modal dialog, or switch the
    dialog to non-modal, you MUST refactor to write into a private
    `DeconvResults` and merge into `proc.foopsi` on the success signal.
    """
    progress    = pyqtSignal(int, int)   # (done, total)
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(int)        # n_succeeded
    failed      = pyqtSignal(str, str)

    def __init__(self, est, proc, ops, cells):
        super().__init__()
        self.est, self.proc, self.ops, self.cells = est, proc, ops, cells

    def run(self) -> None:
        try:
            from caiman_sorter_py.core.deconvolution import run_foopsi
            n_ok = run_foopsi(
                self.est, self.proc, self.ops, cells=self.cells,
                log_cb=lambda m: self.log.emit(m),
                progress_cb=lambda d, t: self.progress.emit(d, t),
                parallel=True,
            )
            self.finished_ok.emit(int(n_ok))
        except Exception as exc:
            import traceback
            self.failed.emit(str(exc), traceback.format_exc())


class _FoopsiDialog(QDialog):
    """Non-closable modal dialog with progress bar shown while foopsi runs."""

    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deconvolving")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title = QLabel("Constrained foopsi / OASIS")
        f = title.font(); f.setPointSize(11); f.setBold(True); title.setFont(f)
        layout.addWidget(title)

        plural = "cell" if total == 1 else "cells"
        self._status = QLabel(f"Deconvolving {total} {plural}…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, max(total, 1))
        layout.addWidget(self._bar)

    def set_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(total, 1))
        self._bar.setValue(done)
        plural = "cell" if total == 1 else "cells"
        self._status.setText(f"Deconvolved {done} / {total} {plural}…")

    def closeEvent(self, event) -> None:
        event.ignore()


class _LoadingDialog(QDialog):
    """Non-closable modal dialog shown while data loads or saves."""

    def __init__(self, parent=None,
                 title: str = "Loading Data",
                 heading: str = "Loading Data…",
                 initial_status: str = "Initializing…"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(
            Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint
        )
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title_lbl = QLabel(heading)
        font  = title_lbl.font()
        font.setPointSize(11)
        font.setBold(True)
        title_lbl.setFont(font)
        layout.addWidget(title_lbl)

        self._status = QLabel(initial_status)
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(40)
        layout.addWidget(self._status)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def closeEvent(self, event) -> None:
        event.ignore()   # prevent user from closing while busy


class _SaveWorker(QThread):
    """Runs session save (and optional .mat export) off the main thread.

    Only READS from `est/proc/ops` — never mutates them. Even if the main
    thread changed those objects mid-save (which the modal `_LoadingDialog`
    blocks), the worst case is an inconsistent on-disk snapshot, not corruption.
    """
    progress  = pyqtSignal(str)
    succeeded = pyqtSignal(str, str)   # h5 path, mat path ('' if none)
    failed    = pyqtSignal(str, str, str)  # message, traceback, which ('h5'|'mat')

    def __init__(self, est, proc, ops, h5_path: str,
                 mat_path: str, source_path: str):
        super().__init__()
        self.est, self.proc, self.ops = est, proc, ops
        self.h5_path = h5_path
        self.mat_path = mat_path
        self.source_path = source_path

    def run(self) -> None:
        try:
            from caiman_sorter_py.io.session import save_session
            self.progress.emit(f"Writing session HDF5 ({Path(self.h5_path).name})…")
            save_session(self.h5_path, self.est, self.proc, self.ops,
                         source_path=self.source_path)
        except Exception as exc:
            import traceback
            self.failed.emit(str(exc), traceback.format_exc(), "h5")
            return

        if self.mat_path:
            try:
                from caiman_sorter_py.io.mat_export import save_session_mat
                self.progress.emit(f"Writing MATLAB v7.3 ({Path(self.mat_path).name})…")
                save_session_mat(self.mat_path, self.est, self.proc, self.ops,
                                 source_path=self.source_path)
            except Exception as exc:
                import traceback
                self.failed.emit(str(exc), traceback.format_exc(), "mat")
                return

        self.succeeded.emit(self.h5_path, self.mat_path)


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self.session = Session()
        # Path of the file behind the currently loaded session — cached so save
        # dialogs can derive a default filename even after the load worker
        # QThread has been dropped.
        self._loaded_path: str = ""
        self.setWindowTitle(f"CaImAn Sorter v{__version__}")
        self.resize(1600, 950)
        self._build_ui()
        self._connect_signals()
        self._restore_settings()
        self.log(f"CaImAn Sorter v{__version__} started.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        root.addLayout(self._build_toolbar())
        root.addWidget(self._build_main_splitter(), stretch=1)
        root.addWidget(self._build_log())

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setToolTip(
            "Pick a file to open. Accepts:\n"
            "  • CaImAn HDF5 output (*.hdf5, *.h5)\n"
            "  • Saved sort session (*.h5)\n"
            "  • Saved sort session (*.mat — MATLAB v7.3)\n"
            "  • Saved ops only (*_ops.h5) — applies params without touching data"
        )
        self.filepath_edit = QLineEdit()
        self.filepath_edit.setPlaceholderText("Select a CaImAn .hdf5 / session .h5 / sort .mat file…")
        self.filepath_edit.setReadOnly(True)
        self.filepath_edit.setToolTip("Path of the currently selected file (read-only).")
        self.load_btn = QPushButton("Load")
        self.load_btn.setToolTip(
            "Load the selected file. Auto-detects format:\n"
            "  • CaImAn HDF5 → runs full proc init (~5–30 s)\n"
            "  • Sort session (.h5 / .mat) → restores est/proc/ops directly (fast)\n"
            "  • Ops file → applies params only, no reload"
        )
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(
            "Save the full sort session as .h5 (self-contained).\n"
            "If 'Also save MATLAB .mat' is on in the Params tab, a .mat sidecar\n"
            "is written too, matching the legacy MATLAB pipeline format."
        )
        self.save_ops_btn = QPushButton("Save Ops")
        self.save_ops_btn.setEnabled(False)
        self.save_ops_btn.setToolTip(
            "Save only the current params (evaluation thresholds, deconv settings,\n"
            "smoothing, save tag, etc.) as a small _ops.h5 file.\n"
            "Useful for sharing parameters between sessions or recordings."
        )

        row.addWidget(self.browse_btn)
        row.addWidget(self.filepath_edit, stretch=1)
        row.addWidget(self.load_btn)
        row.addWidget(self.save_btn)
        row.addWidget(self.save_ops_btn)
        return row

    def _build_main_splitter(self) -> QSplitter:
        self.main_splitter = QSplitter(Qt.Horizontal)

        self.image_panel = ImagePanel(self.session)
        self.trace_panel = TracePanel(self.session)
        self.params_panel = ParamsPanel(self.session)
        self.nav_panel = NavPanel(self.session)
        self.merge_panel = MergePanel(self.session)
        from caiman_sorter_py.ui.batch_panel import BatchPanel
        self.batch_panel = BatchPanel(self.session)

        # Center: tab widget (images / deconvolution / merge / params / batch)
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.image_panel, "Images")
        self.center_tabs.addTab(self.params_panel.build_deconv_panel(), "Deconvolution")
        self.center_tabs.addTab(self.merge_panel, "Merge")
        self.center_tabs.addTab(self._build_params_tab(), "Params")
        self.center_tabs.addTab(self.batch_panel, "Batch")

        # Left area: tabs on top, trace always visible below
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.addWidget(self.center_tabs)
        self.left_splitter.addWidget(self.trace_panel)
        self.left_splitter.setSizes([580, 280])

        # Right: component image (top) → tabbed eval+metrics (middle) → cell nav (bottom)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.params_panel, "Evaluation")
        self.right_tabs.addTab(self.nav_panel.metrics_box, "Cell Metrics")

        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.addWidget(self.nav_panel.component_box)
        self.right_splitter.addWidget(self.right_tabs)
        self.right_splitter.addWidget(self.nav_panel)
        self.right_splitter.setSizes([160, 460, 200])

        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizes([1280, 320])
        return self.main_splitter

    def _build_params_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        load_group = QGroupBox("Load settings")
        lg = QVBoxLayout(load_group)
        self.load_rejected_chk = QCheckBox("Include CaImAn-rejected components")
        self.load_rejected_chk.setChecked(self.session.ops.load_caiman_rejected)
        self.load_rejected_chk.setToolTip(
            "When checked, the discarded_components group is also loaded,\n"
            "giving access to all components CaImAn originally rejected.\n"
            "Takes effect on the next Load."
        )
        self.load_rejected_chk.stateChanged.connect(
            lambda state: setattr(self.session.ops, "load_caiman_rejected", bool(state))
        )
        lg.addWidget(self.load_rejected_chk)
        v.addWidget(load_group)

        save_group = QGroupBox("Save settings")
        sg = QVBoxLayout(save_group)

        tag_row = QHBoxLayout()
        tag_row.addWidget(QLabel("Save tag:"))
        self.save_tag_edit = QLineEdit(self.session.ops.save_tag)
        self.save_tag_edit.setToolTip(
            "String appended to the source stem when generating default save filenames.\n"
            "E.g. with tag '_sort' and source 'M1_results_cnmf.hdf5' the default save names are:\n"
            "  M1_results_cnmf_sort.h5  and  M1_results_cnmf_sort.mat\n"
            "If the source stem already ends with this tag (e.g. you re-opened\n"
            "M1_results_cnmf_sort.h5), the tag is not appended again."
        )
        self.save_tag_edit.textChanged.connect(
            lambda txt: setattr(self.session.ops, "save_tag", txt)
        )
        tag_row.addWidget(self.save_tag_edit, stretch=1)
        sg.addLayout(tag_row)

        self.save_as_mat_chk = QCheckBox("Also save MATLAB v7.3 .mat (for legacy MATLAB code)")
        self.save_as_mat_chk.setChecked(self.session.ops.save_as_mat)
        self.save_as_mat_chk.setToolTip(
            "When checked, the Save button writes a MATLAB .mat sidecar with the same\n"
            "est/proc/ops layout as f_cs_save_data.m (the legacy MATLAB pipeline),\n"
            "so downstream MATLAB analysis scripts can read it directly."
        )
        self.save_as_mat_chk.stateChanged.connect(
            lambda state: setattr(self.session.ops, "save_as_mat", bool(state))
        )
        sg.addWidget(self.save_as_mat_chk)
        v.addWidget(save_group)

        contour_group = QGroupBox("Contour settings")
        cf = QFormLayout(contour_group)
        self.contour_thr_spin = QDoubleSpinBox()
        self.contour_thr_spin.setRange(0.0001, 0.99)
        self.contour_thr_spin.setSingleStep(0.005)
        self.contour_thr_spin.setDecimals(4)
        self.contour_thr_spin.setValue(self.session.ops.contour_thr)
        self.contour_thr_spin.setFixedWidth(90)
        self.contour_thr_spin.setToolTip(
            "Amplitude threshold for contour tracing, as a fraction of each\n"
            "cell's peak footprint value. Lower values trace further out.\n"
            "Click 'Recompute Contours' to apply without reloading."
        )
        self.contour_thr_spin.valueChanged.connect(
            lambda v: setattr(self.session.ops, "contour_thr", v)
        )
        cf.addRow("Threshold:", self.contour_thr_spin)

        self.recompute_contours_btn = QPushButton("Recompute Contours")
        self.recompute_contours_btn.setEnabled(False)
        self.recompute_contours_btn.setToolTip(
            "Recompute contours from the loaded data using the current threshold.\n"
            "Faster than reloading the file."
        )
        self.recompute_contours_btn.clicked.connect(self._on_recompute_contours)
        cf.addRow(self.recompute_contours_btn)
        v.addWidget(contour_group)

        # Plot orientation — GUI-only display tweaks (rotation + flips) that
        # persist in QSettings but never get written to data files.
        v.addWidget(self.params_panel.build_plot_group())

        # Reset actions — destructive session-wide buttons live here, not in
        # the right-side eval panel. ParamsPanel owns the widgets + signals.
        v.addWidget(self.params_panel.build_reset_group())

        v.addStretch()
        return w

    def _on_recompute_contours(self) -> None:
        """Recompute contours from est.A with the current threshold and refresh images."""
        est = self.session.est
        if est is None:
            return
        from caiman_sorter_py.core.contours import compute_contours
        est.contours = compute_contours(est.A, est.dims, thr=self.session.ops.contour_thr)
        self.image_panel.refresh_images()

    def _build_log(self) -> QTextEdit:
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(90)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.log_view.setToolTip(
            "Activity log. Auto-scrolls to the bottom when new messages arrive,\n"
            "unless you've manually scrolled up — then it stays where you left it."
        )
        return self.log_view

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.browse_btn.clicked.connect(self._on_browse)
        self.load_btn.clicked.connect(self._on_load)
        self.save_btn.clicked.connect(self._on_save)
        self.save_ops_btn.clicked.connect(self._on_save_ops)

        self.session.add_listener("data_loaded", self._on_data_loaded)

    # ------------------------------------------------------------------
    # Toolbar callbacks
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        start_dir = self._settings.value("last_browse_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CaImAn or session file", start_dir,
            "Readable files (*.hdf5 *.h5 *.mat);;"
            "CaImAn / Session HDF5 (*.hdf5 *.h5);;"
            "Sort .mat (*.mat);;"
            "Ops file (*_ops.hdf5 *_ops.h5);;"
            "All files (*)"
        )
        if path:
            self.filepath_edit.setText(path)
            self._settings.setValue("last_browse_dir", str(Path(path).parent))
            self.log(f"Selected: {path}")

    def _on_load(self) -> None:
        path = self.filepath_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please browse to a file first.")
            return

        # Ops-only files just update params; no worker / dialog needed.
        from caiman_sorter_py.io.session import detect_format
        fmt = detect_format(path)
        if fmt == "ops":
            try:
                from caiman_sorter_py.io.session import load_ops
                loaded = load_ops(path)
                # _ops.h5 deliberately doesn't persist save_tag / save_as_mat
                # / browse_path / ops_path (per-user prefs, not session state).
                # Preserve the user's current values for those across a reload
                # so the operation only overwrites the params the file owns.
                prev = self.session.ops
                loaded.save_tag    = prev.save_tag
                loaded.save_as_mat = prev.save_as_mat
                loaded.browse_path = prev.browse_path
                loaded.ops_path    = prev.ops_path
                self.session.ops   = loaded
                self.params_panel.load_ops()
                if hasattr(self, "contour_thr_spin"):
                    self.contour_thr_spin.setValue(self.session.ops.contour_thr)
                if hasattr(self, "load_rejected_chk"):
                    self.load_rejected_chk.setChecked(self.session.ops.load_caiman_rejected)
            except Exception as exc:
                QMessageBox.critical(self, "Load Ops error", str(exc))
                return
            self.log(f"Ops loaded from: {path}")
            return

        self.log(f"Loading: {path} …")
        self.load_btn.setEnabled(False)

        self._loading_dlg = _LoadingDialog(self)

        self._load_worker = _LoadWorker(path, self.session.ops)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.succeeded.connect(self._on_load_succeeded)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()

        self._loading_dlg.exec_()   # blocks main thread only via event loop

    def _on_load_progress(self, msg: str) -> None:
        self.log(msg)
        self._loading_dlg.set_status(msg)

    def _on_load_succeeded(self, est, proc, loaded_ops) -> None:
        # Close the dialog and re-enable controls FIRST so a downstream
        # exception (e.g. inside a panel's data_loaded listener) can't leave
        # the modal stuck on screen.
        try:
            if loaded_ops is not None:
                # Restoring a saved session — adopt the saved ops in place.
                self.session.ops = loaded_ops
                self.params_panel.load_ops()
                if hasattr(self, "contour_thr_spin"):
                    self.contour_thr_spin.setValue(loaded_ops.contour_thr)
                if hasattr(self, "load_rejected_chk"):
                    self.load_rejected_chk.setChecked(loaded_ops.load_caiman_rejected)
            self.session.load_data(est, proc)
            self._loaded_path = self._load_worker.path
            self._settings.setValue("last_file", self._loaded_path)
            n_acc = int(proc.accepted.sum())
            self.log(f"Ready: {proc.num_cells} cells ({n_acc} accepted, "
                     f"{proc.num_cells - n_acc} rejected), dims={est.dims}.")
        except Exception as exc:
            import traceback
            self.log(f"Post-load error: {exc}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Post-load error", str(exc))
        finally:
            self._loading_dlg.done(0)
            self.load_btn.setEnabled(True)
            # Drop the worker QThread (and its refs to est/proc/ops) — it has
            # finished and we shouldn't pin those payloads in memory.
            self._load_worker = None

    def _on_load_failed(self, msg: str, tb: str) -> None:
        try:
            self.log(f"Load error: {msg}")
            self.log(tb)
            QMessageBox.critical(self, "Load error", msg)
        finally:
            self._loading_dlg.done(0)
            self.load_btn.setEnabled(True)
            self._load_worker = None

    # ------------------------------------------------------------------
    # Deconvolution (foopsi) async runner
    # ------------------------------------------------------------------

    def run_foopsi_async(self, cells) -> None:
        """Run constrained foopsi for many cells off-thread, with progress dialog.

        Called from ParamsPanel when the user clicks 'Run foopsi' and the
        'Current cell only' checkbox is unchecked.
        """
        if self.session.est is None:
            return
        total = int(len(cells))
        self.log(f"Running foopsi on {total} cells…")
        self.params_panel.run_foopsi_btn.setEnabled(False)

        self._foopsi_dlg = _FoopsiDialog(total, self)
        self._foopsi_worker = _FoopsiWorker(
            self.session.est, self.session.proc, self.session.ops, cells,
        )
        self._foopsi_worker.progress.connect(self._foopsi_dlg.set_progress)
        self._foopsi_worker.log.connect(self.log)
        self._foopsi_worker.finished_ok.connect(self._on_foopsi_done)
        self._foopsi_worker.failed.connect(self._on_foopsi_failed)
        self._foopsi_worker.start()
        self._foopsi_dlg.exec_()

    def _on_foopsi_done(self, n_ok: int) -> None:
        try:
            self.session.refresh_cell()
        except Exception as exc:
            import traceback
            self.log(f"Post-foopsi refresh error: {exc}")
            self.log(traceback.format_exc())
        finally:
            self._foopsi_dlg.done(0)
            self.params_panel.run_foopsi_btn.setEnabled(True)
            self._foopsi_worker = None

    def _on_foopsi_failed(self, msg: str, tb: str) -> None:
        try:
            self.log(f"foopsi error: {msg}")
            self.log(tb)
            QMessageBox.critical(self, "foopsi error", msg)
        finally:
            self._foopsi_dlg.done(0)
            self.params_panel.run_foopsi_btn.setEnabled(True)
            self._foopsi_worker = None

    def _on_save(self) -> None:
        """Save full session to an .h5 file (self-contained)."""
        if self.session.est is None or self.session.proc is None:
            QMessageBox.warning(self, "No data", "Load a file before saving.")
            return
        # Sync UI controls into ops so what we save matches what's on screen
        self.params_panel.sync_to_ops()

        source = self._loaded_path
        tag = self.session.ops.save_tag or ""
        default = self._default_save_path(source, tag, ".h5")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", default,
            "Sort session HDF5 (*.h5 *.hdf5);;Sort .mat (*.mat);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith((".h5", ".hdf5")):
            path += ".h5"

        # Refresh shaped (S_proc) traces from current params before writing (no
        # run button, so this bakes in on-screen edits). smooth dF/dt is fully
        # recomputed; foopsi S_proc is re-derived from the stored raw spikes
        # (cheap, no solver re-run). After path commit so a cancelled save skips
        # the work.
        from caiman_sorter_py.core.deconvolution import (
            refresh_foopsi_proc, run_smooth_dfdt,
        )
        run_smooth_dfdt(self.session.est, self.session.proc, self.session.ops,
                        log_cb=self.log)
        refresh_foopsi_proc(self.session.est, self.session.proc, self.session.ops)

        mat_path = self._mat_sidecar_path(path) if self.session.ops.save_as_mat else ""

        self.log(f"Saving: {path} …")
        if mat_path:
            self.log(f"  + MATLAB .mat sidecar: {mat_path}")
        self.save_btn.setEnabled(False)
        self.save_ops_btn.setEnabled(False)

        self._saving_dlg = _LoadingDialog(
            self,
            title="Saving Data",
            heading="Saving Data…",
            initial_status="Preparing…",
        )
        self._save_worker = _SaveWorker(
            self.session.est, self.session.proc, self.session.ops,
            path, mat_path, source,
        )
        self._save_worker.progress.connect(self._on_save_progress)
        self._save_worker.succeeded.connect(self._on_save_succeeded)
        self._save_worker.failed.connect(self._on_save_failed)
        self._save_worker.start()
        self._saving_dlg.exec_()

    def _on_save_progress(self, msg: str) -> None:
        self.log(msg)
        if hasattr(self, "_saving_dlg"):
            self._saving_dlg.set_status(msg)

    def _on_save_succeeded(self, h5_path: str, mat_path: str) -> None:
        try:
            self.log(f"Session saved: {h5_path}")
            if mat_path:
                self.log(f"MATLAB .mat saved: {mat_path}")
        finally:
            self._saving_dlg.done(0)
            self.save_btn.setEnabled(True)
            self.save_ops_btn.setEnabled(True)
            self._save_worker = None

    def _on_save_failed(self, msg: str, tb: str, which: str) -> None:
        try:
            self.log(f"Save error ({which}): {msg}")
            self.log(tb)
            if which == "mat":
                QMessageBox.warning(self, ".mat save error",
                                    f"Session HDF5 was written, but .mat export failed:\n\n{msg}")
            else:
                QMessageBox.critical(self, "Save error", msg)
        finally:
            self._saving_dlg.done(0)
            self.save_btn.setEnabled(True)
            self.save_ops_btn.setEnabled(True)
            self._save_worker = None

    def _mat_sidecar_path(self, h5_path: str) -> str:
        """Derive the .mat sidecar path from the chosen .h5 save path.

        Just swaps the extension to .mat — whatever tag the user baked into
        the .h5 filename is preserved in the stem.
        """
        p = Path(h5_path)
        return str(p.with_suffix(".mat"))

    def _on_save_ops(self) -> None:
        """Save the ops/parameters to a small _ops.h5 file."""
        self.params_panel.sync_to_ops()
        source = self._loaded_path
        tag = self.session.ops.save_tag or ""
        default = self._default_save_path(source, tag, "_ops.h5")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ops", default,
            "Ops file (*_ops.h5 *_ops.hdf5);;HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith((".h5", ".hdf5")):
            path += "_ops.h5"
        try:
            from caiman_sorter_py.io.session import save_ops
            save_ops(path, self.session.ops)
        except Exception as exc:
            import traceback
            self.log(f"Save Ops error: {exc}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Save Ops error", str(exc))
            return
        self.log(f"Ops saved: {path}")

    def _default_save_path(self, source: str, tag: str, extra: str) -> str:
        """Pick a default save path: <source_stem><tag><extra>.

        Skips `tag` when the source stem already ends with it, so re-saving a
        previously-sorted file (e.g. `M1_results_cnmf_sort.h5`) defaults to
        the same filename instead of accumulating `_sort_sort…` suffixes.
        """
        if not source:
            return ""
        p = Path(source)
        stem = p.stem
        if tag and stem.endswith(tag):
            return str(p.parent / (stem + extra))
        return str(p.parent / (stem + tag + extra))

    def _on_data_loaded(self) -> None:
        self.save_btn.setEnabled(True)
        self.save_ops_btn.setEnabled(True)
        self.recompute_contours_btn.setEnabled(True)
        n = self.session.proc.num_cells
        self.log(f"Loaded {n} cells.")

    # ------------------------------------------------------------------
    # Persistent settings
    # ------------------------------------------------------------------

    def _restore_settings(self) -> None:
        """Restore window geometry, splitter sizes, last file path, and ops."""
        geom = self._settings.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)

        main_sizes = self._settings.value("window/splitter_main_v2")
        if main_sizes:
            self.main_splitter.restoreState(main_sizes)

        left_sizes = self._settings.value("window/splitter_left_v1")
        if left_sizes:
            self.left_splitter.restoreState(left_sizes)

        right_sizes = self._settings.value("window/splitter_right_v2")
        if right_sizes:
            self.right_splitter.restoreState(right_sizes)

        last_file = self._settings.value("last_file", "")
        if last_file:
            self.filepath_edit.setText(last_file)

        self._restore_ops_settings()
        self.params_panel.load_ops()
        self.batch_panel.load_ops()

    def closeEvent(self, event) -> None:
        """Save window geometry, splitter sizes, and ops on close.

        Refuse to close while a worker QThread is still running — destroying the
        thread mid-run would corrupt whatever it's doing (save half-written file,
        deconv partially populated, etc.) and Qt would log
        "QThread: Destroyed while thread is still running" before potentially
        crashing.
        """
        running = self._running_workers()
        if running:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Background task running",
                f"A background task is still running: {', '.join(running)}.\n\n"
                "Wait for it to finish, or quit immediately and lose its work?",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Abort,
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # User chose Abort — block briefly waiting for the workers, then quit.
            # Batch worker has a cooperative stop flag (no event loop to .quit())
            # — set it explicitly then wait the run() method out.
            bw = self._batch_worker_or_none()
            if bw is not None:
                bw.stop()
            for w in (self._load_worker_or_none(),
                      self._save_worker_or_none(),
                      self._foopsi_worker_or_none(),
                      bw):
                if w is not None and w.isRunning():
                    w.quit()
                    w.wait(3000)   # ms — give threads a moment to wind down

        self.params_panel.sync_to_ops()
        # Belt-and-braces: read MainWindow-owned widgets directly into ops so
        # an unfocused-but-edited field still gets persisted.
        self.session.ops.save_tag = self.save_tag_edit.text()
        self._save_ops_settings()
        self._settings.setValue("window/geometry",          self.saveGeometry())
        self._settings.setValue("window/splitter_main_v2",  self.main_splitter.saveState())
        self._settings.setValue("window/splitter_left_v1",  self.left_splitter.saveState())
        self._settings.setValue("window/splitter_right_v2", self.right_splitter.saveState())
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Worker bookkeeping
    # ------------------------------------------------------------------

    def _load_worker_or_none(self) -> "QThread | None":
        w = getattr(self, "_load_worker", None)
        return w if (w is not None and w.isRunning()) else None

    def _save_worker_or_none(self) -> "QThread | None":
        w = getattr(self, "_save_worker", None)
        return w if (w is not None and w.isRunning()) else None

    def _foopsi_worker_or_none(self) -> "QThread | None":
        w = getattr(self, "_foopsi_worker", None)
        return w if (w is not None and w.isRunning()) else None

    def _batch_worker_or_none(self) -> "QThread | None":
        w = getattr(self.batch_panel, "_worker", None)
        return w if (w is not None and w.isRunning()) else None

    def _running_workers(self) -> list[str]:
        out = []
        if self._load_worker_or_none()   is not None: out.append("Load")
        if self._save_worker_or_none()   is not None: out.append("Save")
        if self._foopsi_worker_or_none() is not None: out.append("foopsi")
        if self._batch_worker_or_none()  is not None: out.append("Batch")
        return out

    def _save_ops_settings(self) -> None:
        """Persist every Ops field to QSettings.

        Iterates `OPS_TOP_FIELD_NAMES`, `OPS_SUB_PREFIXES`, AND
        `QSETTINGS_ONLY_SUBS`. The last group (currently just `plot`) is
        QSettings-only — kept out of data files but still persisted across
        app launches.
        """
        s   = self._settings
        ops = self.session.ops

        for name in OPS_TOP_FIELD_NAMES:
            s.setValue(f"ops/{name}", getattr(ops, name))

        for attr_name, prefix in {**OPS_SUB_PREFIXES, **QSETTINGS_ONLY_SUBS}.items():
            sub = getattr(ops, attr_name)
            for fld in fields(sub):
                s.setValue(f"ops/{prefix}/{fld.name}", getattr(sub, fld.name))

    def _restore_ops_settings(self) -> None:
        """Load persisted Ops fields from QSettings.

        Driven by `OPS_SUB_PREFIXES` and `OPS_TOP_FIELD_NAMES` in core/state.
        Each field's existing default value drives the type coercion — bool
        defaults coerce strings via the truthy-token check, numeric defaults
        use int()/float(), string defaults use str().
        """
        s   = self._settings
        ops = self.session.ops

        def _coerce(raw, default):
            if raw is None:
                return default
            # QSettings returns strings on read; coerce by the default's type.
            if isinstance(default, bool):
                return raw in (True, "true", "1", 1)
            if isinstance(default, int) and not isinstance(default, bool):
                return int(raw)
            if isinstance(default, float):
                return float(raw)
            return str(raw)

        for name in OPS_TOP_FIELD_NAMES:
            default = getattr(ops, name)
            setattr(ops, name, _coerce(s.value(f"ops/{name}"), default))

        for attr_name, prefix in {**OPS_SUB_PREFIXES, **QSETTINGS_ONLY_SUBS}.items():
            sub = getattr(ops, attr_name)
            for fld in fields(sub):
                default = getattr(sub, fld.name)
                setattr(sub, fld.name,
                        _coerce(s.value(f"ops/{prefix}/{fld.name}"), default))

        # Sync UI widgets that mirror specific top-level ops fields.
        if hasattr(self, "load_rejected_chk"):
            self.load_rejected_chk.setChecked(ops.load_caiman_rejected)
        if hasattr(self, "save_tag_edit"):
            self.save_tag_edit.setText(ops.save_tag)
        if hasattr(self, "save_as_mat_chk"):
            self.save_as_mat_chk.setChecked(ops.save_as_mat)
        if hasattr(self, "contour_thr_spin"):
            self.contour_thr_spin.setValue(ops.contour_thr)
        if hasattr(self, "merge_panel"):
            self.merge_panel.load_ops()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        sb = self.log_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        prev = sb.value()
        self.log_view.append(f"[{ts}] {message}")
        if not at_bottom:
            sb.setValue(prev)   # preserve scroll position if user has scrolled up
