"""Batch processing tab — apply the current GUI ops to every CaImAn HDF5
in a chosen directory, with optional eval / merge / smooth dF/dt / foopsi steps.

Runs on a QThread so the GUI stays responsive. Each file is loaded in
isolation; the main session (est/proc/ops in the GUI) is never touched.
The ops template is deep-copied at run start, so further GUI edits during
the batch don't change what the worker is using.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


def _safe_ops_dump(ops) -> dict:
    """Convert an Ops dataclass to a JSON-safe dict.

    `dataclasses.asdict` handles nested dataclasses; this wrapper coerces
    numpy scalars (int/float) and Paths to native Python types so json.dumps
    doesn't choke. Anything stranger falls back to its repr via the
    `default=str` parameter on json.dumps.
    """
    import numpy as np
    raw = asdict(ops)
    def _coerce(v):
        if isinstance(v, dict):
            return {k: _coerce(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        return v
    return _coerce(raw)


# Filename suffixes we glob for as inputs.
INPUT_EXTS = (".hdf5", ".h5", ".hdf")
# Suffixes we always SKIP — these are outputs of a prior run, not inputs.
OUTPUT_SUFFIX_MARKERS = ("_sort.", "_ops.")


@dataclass
class _BatchSteps:
    """Which pipeline steps the user wants for this run."""
    evaluate: bool = True
    merge: bool = True
    smooth_dfdt: bool = False
    foopsi: bool = False


class _BatchWorker(QThread):
    """Background runner. Emits progress + finished signals."""

    progress = pyqtSignal(str)                       # log line
    finished_all = pyqtSignal(int, int, int, int)    # processed, skipped, failed, total

    def __init__(self, file_paths: list, ops_template, steps: _BatchSteps,
                 overwrite: bool, parent=None):
        super().__init__(parent)
        self._paths = list(file_paths)
        # Deep copy so live GUI edits during batch don't move the goalposts.
        self._ops = copy.deepcopy(ops_template)
        self._steps = steps
        self._overwrite = overwrite
        self._stop = False
        self._results: list[dict] = []  # per-file records, written into the JSON report

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        processed = skipped = failed = 0
        total = len(self._paths)
        t0 = time.time()
        run_start_ts = _dt.datetime.now()
        for idx, path in enumerate(self._paths, start=1):
            if self._stop:
                self.progress.emit(f"[STOP] aborted after {idx-1}/{total}")
                break
            label = f"{idx}/{total} {Path(path).name}"
            file_t0 = time.time()
            rec: dict = {"input": str(path), "name": Path(path).name}
            try:
                status = self._process_one(path, label, rec)
                rec["status"] = status
                if status == "processed":
                    processed += 1
                elif status == "skipped":
                    skipped += 1
            except Exception as exc:
                failed += 1
                rec["status"] = "failed"
                rec["error"] = repr(exc)
                self.progress.emit(f"{label} — FAILED: {exc}")
            rec["duration_s"] = round(time.time() - file_t0, 2)
            self._results.append(rec)
        elapsed = time.time() - t0
        self.progress.emit(
            f"[DONE] {processed} processed · {skipped} skipped · {failed} failed "
            f"({elapsed:.1f}s total)"
        )
        report_path = self._write_report(run_start_ts, elapsed,
                                         processed, skipped, failed, total)
        if report_path is not None:
            self.progress.emit(f"[REPORT] {report_path}")
        self.finished_all.emit(processed, skipped, failed, total)

    def _write_report(self, started_at: _dt.datetime, elapsed: float,
                      processed: int, skipped: int, failed: int,
                      total: int) -> Optional[Path]:
        """Dump a JSON record of this run + the ops snapshot + per-file results.

        Lands in the input directory (parent of the first input file). Filename
        embeds the start timestamp so concurrent / repeated runs don't clobber
        each other.
        """
        if not self._paths:
            return None
        out_dir = Path(self._paths[0]).parent
        stamp = started_at.strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
        out_path = out_dir / f"batch_log_{stamp}.json"
        try:
            payload = {
                "run": {
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "duration_s": round(elapsed, 2),
                    "input_dir": str(out_dir),
                    "n_total": total,
                    "n_processed": processed,
                    "n_skipped": skipped,
                    "n_failed": failed,
                    "steps": asdict(self._steps),
                    "overwrite": self._overwrite,
                },
                "ops": _safe_ops_dump(self._ops),
                "files": self._results,
            }
            out_path.write_text(json.dumps(payload, indent=2, default=str),
                                encoding="utf-8")
            return out_path
        except Exception as exc:
            self.progress.emit(f"[REPORT] failed to write: {exc}")
            return None

    # ------------------------------------------------------------------
    # Per-file pipeline
    # ------------------------------------------------------------------

    def _process_one(self, path: str, label: str, rec: dict) -> str:
        from caiman_sorter_py.io.session import detect_format, load_session, save_session
        from caiman_sorter_py.io.hdf5_loader import load_hdf5
        from caiman_sorter_py.io.mat_loader import load_session_mat
        from caiman_sorter_py.io.mat_export import save_session_mat
        from caiman_sorter_py.core.proc_init import initialize_proc
        from caiman_sorter_py.core.evaluation import evaluate_components, update_accepted
        from caiman_sorter_py.core.merge import (
            find_duplicate_pairs, apply_choose_best_snr, apply_create_new,
        )
        from caiman_sorter_py.core.deconvolution import run_foopsi, run_smooth_dfdt

        ops = self._ops    # template; the worker doesn't mutate session state

        # Compute output path = stem + save_tag + ".h5"
        p = Path(path)
        tag = ops.save_tag or ""
        out_h5 = p.parent / (p.stem + tag + ".h5") if not p.stem.endswith(tag) \
                 else p.parent / (p.stem + ".h5")
        out_mat = out_h5.with_suffix(".mat")

        rec["output_h5"] = str(out_h5)

        if out_h5.exists() and not self._overwrite:
            self.progress.emit(f"{label} — output exists, skip")
            rec["reason"] = "output exists"
            return "skipped"

        # ---- Load est / proc ----
        fmt = detect_format(path)
        rec["format"] = fmt
        if fmt == "caiman":
            self.progress.emit(f"{label} — loading CaImAn HDF5...")
            est = load_hdf5(path, load_rejected=ops.load_caiman_rejected,
                            contour_thr=ops.contour_thr)
            proc = initialize_proc(est, ops)
        elif fmt == "session":
            self.progress.emit(f"{label} — loading session HDF5...")
            est, proc, _ = load_session(path)
        elif fmt == "session_mat":
            self.progress.emit(f"{label} — loading session .mat...")
            est, proc, _ = load_session_mat(path)
        else:
            self.progress.emit(f"{label} — unrecognized format ({fmt}), skip")
            rec["reason"] = f"unrecognized format ({fmt})"
            return "skipped"

        n_cells_init = proc.num_cells
        rec["n_cells_initial"] = n_cells_init

        # ---- Evaluate ----
        n_merges = 0
        if self._steps.evaluate:
            self.progress.emit(f"{label} — evaluating components...")
            core_mask = evaluate_components(est, proc, ops)
            update_accepted(proc, core_mask)

        # ---- Merge ----
        if self._steps.merge:
            self.progress.emit(f"{label} — finding duplicate merges...")
            pairs = find_duplicate_pairs(est, proc, ops)
            n_merges = len(pairs) if pairs else 0
            if pairs:
                if ops.merge.method == "choose best snr":
                    apply_choose_best_snr(proc, pairs)
                else:
                    apply_create_new(est, proc, ops, pairs)
                self.progress.emit(f"{label} — merged {len(pairs)} pairs")
            else:
                self.progress.emit(f"{label} — no merge pairs found")

        # ---- Smooth dF/dt ----
        # Runs before foopsi so the (potentially) faster, no-model deconv is
        # always available even when foopsi is skipped or fails.
        if self._steps.smooth_dfdt:
            self.progress.emit(f"{label} — running smooth dF/dt...")
            run_smooth_dfdt(est, proc, ops)
            self.progress.emit(f"{label} — smooth dF/dt done ({proc.num_cells} cells)")

        # ---- Foopsi ----
        n_foopsi = 0
        if self._steps.foopsi:
            self.progress.emit(f"{label} — running constrained foopsi...")
            n_foopsi = run_foopsi(est, proc, ops, parallel=True)
            self.progress.emit(f"{label} — foopsi done ({n_foopsi} cells)")

        # ---- Save ----
        self.progress.emit(f"{label} — saving {out_h5.name}...")
        save_session(out_h5, est, proc, ops, source_path=str(path))
        rec["saved_mat"] = False
        if ops.save_as_mat:
            save_session_mat(out_mat, est, proc, ops, source_path=str(path))
            self.progress.emit(f"{label} — also saved {out_mat.name}")
            rec["saved_mat"] = True
            rec["output_mat"] = str(out_mat)

        n_acc = int(proc.accepted.sum()) if proc.accepted is not None else 0
        rec["n_cells_final"] = int(proc.num_cells)
        rec["n_accepted"] = n_acc
        rec["n_merges"] = int(n_merges)
        rec["n_foopsi"] = int(n_foopsi)
        self.progress.emit(
            f"{label} — OK ({n_cells_init} → {proc.num_cells} cells, {n_acc} accepted)"
        )
        return "processed"


class BatchPanel(QWidget):
    """Tab UI for batch processing a directory of CaImAn HDF5 files."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._worker: Optional[_BatchWorker] = None
        self._build_ui()
        # Wire widget changes → sync into ops.batch so the QSettings round-trip
        # picks them up at app close. load_ops() is called from MainWindow after
        # _restore_ops_settings(), pushing persisted values back into the widgets.
        self.dir_edit.textChanged.connect(self._sync_to_ops)
        self.eval_chk.toggled.connect(self._sync_to_ops)
        self.merge_chk.toggled.connect(self._sync_to_ops)
        self.smooth_dfdt_chk.toggled.connect(self._sync_to_ops)
        self.foopsi_chk.toggled.connect(self._sync_to_ops)
        self.overwrite_chk.toggled.connect(self._sync_to_ops)

    def load_ops(self) -> None:
        """Populate widgets from session.ops.batch. Called by MainWindow after
        QSettings restore. blockSignals so we don't fire textChanged/toggled
        and immediately write the same values back."""
        b = self.session.ops.batch
        widgets = (self.dir_edit, self.eval_chk, self.merge_chk,
                   self.smooth_dfdt_chk, self.foopsi_chk, self.overwrite_chk)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.dir_edit.setText(b.input_dir)
            self.eval_chk.setChecked(b.do_evaluate)
            self.merge_chk.setChecked(b.do_merge)
            self.smooth_dfdt_chk.setChecked(b.do_smooth_dfdt)
            self.foopsi_chk.setChecked(b.do_foopsi)
            self.overwrite_chk.setChecked(b.overwrite)
        finally:
            for w in widgets:
                w.blockSignals(False)
        # Refresh the file count label to match the restored directory.
        self._on_refresh()

    def _sync_to_ops(self, *_args) -> None:
        """Push current widget state into session.ops.batch."""
        b = self.session.ops.batch
        b.input_dir = self.dir_edit.text()
        b.do_evaluate = self.eval_chk.isChecked()
        b.do_merge = self.merge_chk.isChecked()
        b.do_smooth_dfdt = self.smooth_dfdt_chk.isChecked()
        b.do_foopsi = self.foopsi_chk.isChecked()
        b.overwrite = self.overwrite_chk.isChecked()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Input directory ----
        dir_group = QGroupBox("Input")
        dg = QVBoxLayout(dir_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Directory:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Pick a folder of CaImAn .hdf5 files…")
        self.dir_edit.textChanged.connect(self._on_dir_changed)
        row.addWidget(self.dir_edit, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse)
        row.addWidget(browse_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh)
        row.addWidget(refresh_btn)
        dg.addLayout(row)

        self.file_count_label = QLabel("No directory selected.")
        self.file_count_label.setStyleSheet("color: gray;")
        dg.addWidget(self.file_count_label)
        layout.addWidget(dir_group)

        # ---- Pipeline steps ----
        steps_group = QGroupBox("Steps")
        sg = QVBoxLayout(steps_group)
        self.eval_chk = QCheckBox("Evaluate components")
        self.eval_chk.setChecked(True)
        self.eval_chk.setToolTip(
            "Run automatic accept/reject using the current GUI eval method + thresholds."
        )
        sg.addWidget(self.eval_chk)
        self.merge_chk = QCheckBox("Find + apply duplicate merges")
        self.merge_chk.setChecked(True)
        self.merge_chk.setToolTip(
            "Detect duplicate pairs and merge them per the Merge tab params."
        )
        sg.addWidget(self.merge_chk)
        self.smooth_dfdt_chk = QCheckBox("Run smooth dF/dt deconvolution")
        self.smooth_dfdt_chk.setChecked(False)
        self.smooth_dfdt_chk.setToolTip(
            "Run smooth dF/dt deconvolution per the Deconvolution tab params. "
            "Fast (no model). Output stored in proc.smooth_dfdt.S."
        )
        sg.addWidget(self.smooth_dfdt_chk)
        self.foopsi_chk = QCheckBox("Run constrained foopsi (slow)")
        self.foopsi_chk.setChecked(False)
        self.foopsi_chk.setToolTip(
            "Run foopsi deconvolution per cell. Adds ~0.5–1 s/cell on OASIS."
        )
        sg.addWidget(self.foopsi_chk)
        layout.addWidget(steps_group)

        # ---- Output options ----
        out_group = QGroupBox("Output")
        og = QVBoxLayout(out_group)
        self.overwrite_chk = QCheckBox("Overwrite existing output")
        self.overwrite_chk.setChecked(False)
        self.overwrite_chk.setToolTip(
            "When unchecked, files that already have a sort output are skipped. "
            "When checked, all files are reprocessed and old outputs overwritten."
        )
        og.addWidget(self.overwrite_chk)
        info = QLabel(
            "Saves: <i>.h5</i> always; <i>.mat</i> sidecar if "
            "<b>Also save MATLAB .mat</b> is enabled in the Params tab."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        og.addWidget(info)
        layout.addWidget(out_group)

        # ---- Settings note + Run/Stop ----
        note = QLabel(
            "Using current GUI settings (eval thresholds, merge params, deconv params)."
        )
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ---- Log ----
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background-color: #1e1e1e; color: #ddd; font-family: Consolas, monospace;"
        )
        layout.addWidget(self.log_view, stretch=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_inputs(self) -> list[Path]:
        """Return CaImAn .hdf5/.h5/.hdf files in the chosen directory.

        Skips outputs of previous runs (anything containing `_sort.` or
        `_ops.` in its name) and anything `detect_format` doesn't recognise
        as a 'caiman' or 'session*' file.
        """
        text = self.dir_edit.text().strip()
        if not text:
            return []
        d = Path(text)
        if not d.is_dir():
            return []
        from caiman_sorter_py.io.session import detect_format
        out: list[Path] = []
        for ext in INPUT_EXTS:
            for p in d.glob(f"*{ext}"):
                if any(m in p.name for m in OUTPUT_SUFFIX_MARKERS):
                    continue
                fmt = detect_format(p)
                if fmt in ("caiman", "session", "session_mat"):
                    out.append(p)
        return sorted(out)

    def _log(self, msg: str) -> None:
        self.log_view.append(msg)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        start = self.dir_edit.text().strip()
        if not start or not Path(start).is_dir():
            start = self.session.ops.batch.input_dir or ""
        d = QFileDialog.getExistingDirectory(self, "Pick input directory", start)
        if d:
            self.dir_edit.setText(d)

    def _on_dir_changed(self, _text: str) -> None:
        self._on_refresh()

    def _on_refresh(self) -> None:
        files = self._gather_inputs()
        if not self.dir_edit.text().strip():
            self.file_count_label.setText("No directory selected.")
        elif not files:
            self.file_count_label.setText(
                "No CaImAn files found (looking for .hdf5/.h5/.hdf, "
                "excluding _sort/_ops outputs)."
            )
        else:
            self.file_count_label.setText(
                f"Found {len(files)} CaImAn file(s) ready to process."
            )

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self.session.ops is None:
            self._log("[ERROR] No ops template — load a file in the main GUI first "
                      "so eval/deconv defaults are populated.")
            return
        files = self._gather_inputs()
        if not files:
            self._log("[ERROR] No input files. Pick a directory and click Refresh.")
            return

        steps = _BatchSteps(
            evaluate=self.eval_chk.isChecked(),
            merge=self.merge_chk.isChecked(),
            smooth_dfdt=self.smooth_dfdt_chk.isChecked(),
            foopsi=self.foopsi_chk.isChecked(),
        )
        self._log(
            f"[START] {len(files)} file(s) · "
            f"evaluate={steps.evaluate} · merge={steps.merge} · "
            f"smooth_dfdt={steps.smooth_dfdt} · foopsi={steps.foopsi} · "
            f"overwrite={self.overwrite_chk.isChecked()}"
        )

        self._worker = _BatchWorker(
            file_paths=[str(p) for p in files],
            ops_template=self.session.ops,
            steps=steps,
            overwrite=self.overwrite_chk.isChecked(),
            parent=self,        # Qt parent → survives Python-side GC races
        )
        self._worker.progress.connect(self._log)
        self._worker.finished_all.connect(self._on_finished_all)
        # QThread's built-in `finished` signal fires AFTER run() returns AND the
        # thread has fully exited. Tie deleteLater + clear-ref to that, not to
        # our custom finished_all signal — emitting finished_all happens inside
        # run()'s last lines, so dropping the Python reference there can race
        # with the thread's cleanup ("QThread: Destroyed while thread is still
        # running").
        self._worker.finished.connect(self._on_thread_finished)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._log("[STOP] requested — finishing current file before exit…")

    def _on_finished_all(self, _processed: int, _skipped: int,
                        _failed: int, _total: int) -> None:
        # Re-enable UI. Don't touch self._worker — the QThread is still in its
        # final cleanup at this moment. _on_thread_finished does the release.
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_thread_finished(self) -> None:
        """QThread has fully exited — now safe to release the C++ object."""
        w = self._worker
        if w is not None:
            w.deleteLater()
        self._worker = None
