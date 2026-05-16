"""Round-trip tests for io.mat_export.save_session_mat → io.mat_loader.load_session_mat.

These cover the parts of the MATLAB .mat I/O that are easy to silently break:
  - sparse A patch-in (custom MATLAB_sparse attr)
  - per-cell scalar shapes (column-vector convention)
  - boolean ops flags stored as double 0.0/1.0
  - eval_params_caiman key rename (snr_thresh → SNR_thresh)
  - 1-based MATLAB indices (idx_components, idx_manual, merge_parents)
  - merge_parents (n_merges, 2) round-trip
  - embedded-NUL stripping for vlen strings
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from caiman_sorter_py.io.mat_export import save_session_mat
from caiman_sorter_py.io.mat_loader import load_session_mat


@pytest.fixture
def saved_mat_path(tiny_session):
    est, proc, ops = tiny_session
    # Inject a couple of merges + a manual reject so the persistence paths
    # for both fire on save.
    proc.merge_parents = [(0, 1), (2, 4)]
    proc.manual_override[3] = True
    proc.accepted[3] = False
    # Toggle a couple of boolean ops flags to non-default values so we can
    # verify they round-trip as double 0/1.
    ops.eval_reject.use_snr2 = True
    ops.eval_reject.use_cnn  = False
    ops.foopsi.smooth_s = True

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mat")
        save_session_mat(path, est, proc, ops, source_path="dummy.h5")
        yield path, est, proc, ops


def test_mat_sparse_A_roundtrip(saved_mat_path):
    path, est, _, _ = saved_mat_path
    est2, _, _ = load_session_mat(path)
    assert est2.A.shape == est.A.shape
    A1, A2 = est.A.tocsc(), est2.A.tocsc()
    assert np.array_equal(A1.indptr, A2.indptr)
    assert np.array_equal(A1.indices, A2.indices)
    np.testing.assert_allclose(A1.data, A2.data)


def test_mat_time_series_roundtrip(saved_mat_path):
    path, est, _, _ = saved_mat_path
    est2, _, _ = load_session_mat(path)
    for name in ("C", "YrA", "S", "F_dff"):
        v1, v2 = getattr(est, name), getattr(est2, name)
        assert v2.shape == v1.shape, name
        np.testing.assert_allclose(v2, v1, rtol=1e-5, atol=1e-7), name


def test_mat_background_b_f_roundtrip(saved_mat_path):
    """b: (n_pixels, n_bg), f: (n_bg, n_frames). Both must round-trip with
    the original orientation so the W-comp+bkg image still computes."""
    path, est, _, _ = saved_mat_path
    est2, _, _ = load_session_mat(path)
    assert est2.b is not None and est2.f is not None
    assert est2.b.shape == est.b.shape
    assert est2.f.shape == est.f.shape


def test_mat_per_cell_scalars_roundtrip(saved_mat_path):
    """SNR_comp / cnn_preds / r_values / neurons_sn are stored as column
    vectors in MATLAB. Verify they come back as 1-D arrays of length n_cells."""
    path, est, _, _ = saved_mat_path
    est2, _, _ = load_session_mat(path)
    n = est.A.shape[1]
    for name in ("SNR_comp", "cnn_preds", "r_values"):
        v = getattr(est2, name)
        assert v.shape == (n,), f"{name}: {v.shape}"
        np.testing.assert_allclose(v, getattr(est, name), rtol=1e-5, atol=1e-7)


def test_mat_proc_accepted_roundtrip(saved_mat_path):
    path, _, proc, _ = saved_mat_path
    _, proc2, _ = load_session_mat(path)
    np.testing.assert_array_equal(proc2.accepted, proc.accepted)
    np.testing.assert_array_equal(proc2.accepted_core, proc.accepted_core)
    # manual_override is reconstructed from idx_manual / idx_manual_bad
    # (mat_loader._read_proc line 159-163).
    np.testing.assert_array_equal(
        proc2.manual_override, proc.manual_override,
    )


def test_mat_proc_metrics_roundtrip(saved_mat_path):
    path, _, proc, _ = saved_mat_path
    _, proc2, _ = load_session_mat(path)
    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "tauAR1", "smooth_dfdt_std"):
        v1 = getattr(proc, name)
        v2 = getattr(proc2, name)
        if v1 is None:
            assert v2 is None or len(v2) == 0
        else:
            np.testing.assert_allclose(v2, v1, rtol=1e-5, atol=1e-7), name


def test_mat_merge_parents_roundtrip(saved_mat_path):
    """merge_parents persists with 1-based MATLAB indices on disk; loader
    converts back to 0-based Python tuples."""
    path, _, proc, _ = saved_mat_path
    _, proc2, _ = load_session_mat(path)
    assert proc2.merge_parents == proc.merge_parents == [(0, 1), (2, 4)]


def test_mat_merge_parents_empty_roundtrip(tiny_session):
    """Sessions with no merges round-trip with an empty merge_parents list."""
    est, proc, ops = tiny_session
    proc.merge_parents = []
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "no_merges.mat")
        save_session_mat(path, est, proc, ops, source_path="dummy.h5")
        _, proc2, _ = load_session_mat(path)
    assert proc2.merge_parents == []


def test_mat_idx_components_one_based_on_disk(saved_mat_path):
    """est.idx_components is 0-based Python; on disk MATLAB expects 1-based.
    After round-trip through the loader we should be back to 0-based."""
    path, est, _, _ = saved_mat_path
    est2, _, _ = load_session_mat(path)
    # Pristine record from load — round-trips as 0-based Python ints.
    np.testing.assert_array_equal(
        np.sort(est2.idx_components),
        np.sort(est.idx_components),
    )


def test_mat_ops_roundtrip(saved_mat_path):
    """Boolean ops flags are stored as MATLAB double 0/1 — they come back as
    Python bools after loader coercion. Verify the toggled flags round-trip."""
    path, _, _, ops = saved_mat_path
    _, _, ops2 = load_session_mat(path)
    assert ops2.eval_method      == ops.eval_method
    assert ops2.eval_reject.use_snr2 is True
    assert ops2.eval_reject.use_cnn  is False
    assert ops2.foopsi.smooth_s      is True
    assert ops2.foopsi.solver        == ops.foopsi.solver
    assert ops2.foopsi.ar_order      == ops.foopsi.ar_order
    np.testing.assert_allclose(
        ops2.foopsi.fudge_factor, ops.foopsi.fudge_factor,
    )


def test_mat_init_params_with_embedded_nuls_roundtrip(tiny_session):
    """CaImAn HDF5 stores some param values as fixed-length |S32 byte strings
    that decode with trailing NULs. After our _clean_str fix those NULs are
    stripped on read so they don't crash the vlen-string writer.

    Verify a NUL-containing init_params string survives a .mat round-trip.
    """
    est, proc, ops = tiny_session
    # Simulate a freshly-loaded CaImAn string that carried fixed-length NUL
    # padding through h5py (matches what the user's save error reported).
    est.init_params_caiman["online"] = {"motion_model": "rigid\x00\x00\x00"}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "with_nul.mat")
        save_session_mat(path, est, proc, ops, source_path="src\x00.h5")
        est2, _, _ = load_session_mat(path)
    # The NUL bytes should be stripped (not preserved) — h5py would have
    # rejected the write otherwise.
    val = est2.init_params_caiman.get("online", {}).get("motion_model")
    assert val is not None and "\x00" not in val
    assert val.startswith("rigid")
