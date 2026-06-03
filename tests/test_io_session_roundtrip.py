"""Round-trip tests for io.session save/load.

Covers the structural invariants that have broken historically:
  - Sparse A (CSC) round-trips with identical data/indices/indptr/shape.
  - est.b and est.f preserve shape (n_pixels, n_bg) and (n_bg, n_frames).
  - float64 dtype is preserved (no silent float32 downcast).
  - Packed deconv format (idx + (k, n_frames) arrays) round-trips, with
    populated cells getting their data back and unpopulated cells staying None.
  - Ops sub-dataclasses (eval_caiman, foopsi, merge, ...) round-trip.
  - init_params_caiman nested dict round-trips.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from caiman_sorter_py.io.session import load_session, save_session


@pytest.fixture
def saved_path(tiny_session):
    est, proc, ops = tiny_session
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.h5")
        save_session(path, est, proc, ops, source_path="dummy.h5")
        yield path, est, proc, ops


def test_sparse_A_roundtrip(saved_path):
    path, est, _, _ = saved_path
    est2, _, _ = load_session(path)
    assert est2.A.shape == est.A.shape
    A1 = est.A.tocsc(); A2 = est2.A.tocsc()
    assert np.array_equal(A1.indptr, A2.indptr)
    assert np.array_equal(A1.indices, A2.indices)
    assert np.allclose(A1.data, A2.data)


def test_time_series_dtype_preserved(saved_path):
    """No silent float32 demotion of C/YrA/S/F_dff."""
    path, _, _, _ = saved_path
    est2, _, _ = load_session(path)
    for name in ("C", "YrA", "S", "F_dff"):
        arr = getattr(est2, name)
        assert arr.dtype == np.float64, f"{name} dtype = {arr.dtype}"


def test_time_series_values(saved_path):
    path, est, _, _ = saved_path
    est2, _, _ = load_session(path)
    assert np.allclose(est2.C,     est.C)
    assert np.allclose(est2.YrA,   est.YrA)
    assert np.allclose(est2.S,     est.S)
    assert np.allclose(est2.F_dff, est.F_dff)


def test_background_b_f_shapes_preserved(saved_path):
    """est.b is (n_pixels, n_bg); est.f is (n_bg, n_frames). Both must round-trip."""
    path, est, _, _ = saved_path
    est2, _, _ = load_session(path)
    assert est2.b is not None and est2.f is not None
    assert est2.b.shape == est.b.shape, f"b: {est2.b.shape} != {est.b.shape}"
    assert est2.f.shape == est.f.shape, f"f: {est2.f.shape} != {est.f.shape}"
    assert np.allclose(est2.b, est.b)
    assert np.allclose(est2.f, est.f)
    # Background image reconstruction (used by ImagePanel "W comp + bkg" mode)
    bkg_img = est2.b @ est2.f.mean(axis=1)
    assert bkg_img.shape == (est.A.shape[0],)


def test_proc_metrics_roundtrip(saved_path):
    path, _, proc, _ = saved_path
    _, proc2, _ = load_session(path)
    assert np.array_equal(proc2.accepted, proc.accepted)
    assert np.array_equal(proc2.accepted_core, proc.accepted_core)
    assert np.array_equal(proc2.manual_override, proc.manual_override)
    for name in ("noise", "skewness", "peaks_ave", "num_zeros",
                 "SNR2_vals", "firing_stab_vals",
                 "gAR1", "gAR2", "tauAR1", "tauAR2", "smooth_dfdt_std"):
        v1 = getattr(proc, name)
        v2 = getattr(proc2, name)
        assert np.allclose(v1, v2), name


def test_packed_deconv_roundtrip(saved_path):
    """Only cell 2 has foopsi data populated. After round-trip:
      - cell 2 keeps its S/C/g exactly,
      - all other cells stay None (not zeros from the dense buffer).
    """
    path, _, proc, _ = saved_path
    _, proc2, _ = load_session(path)
    for i in range(proc.num_cells):
        if proc.foopsi.S[i] is None:
            assert proc2.foopsi.S[i] is None, f"cell {i} S leaked"
            assert proc2.foopsi.S_proc[i] is None, f"cell {i} S_proc leaked"
            assert proc2.foopsi.C[i] is None, f"cell {i} C leaked"
            assert proc2.foopsi.g[i] is None, f"cell {i} g leaked"
        else:
            assert np.allclose(proc2.foopsi.S[i], proc.foopsi.S[i])
            assert np.allclose(proc2.foopsi.S_proc[i], proc.foopsi.S_proc[i])
            assert np.allclose(proc2.foopsi.C[i], proc.foopsi.C[i])
            assert np.allclose(proc2.foopsi.g[i], proc.foopsi.g[i])


def test_init_params_caiman_roundtrip(saved_path):
    path, est, _, _ = saved_path
    est2, _, _ = load_session(path)
    # Both layouts (flat key fr lives under 'data') should remain resolvable
    from caiman_sorter_py.core.state import get_init_param
    assert get_init_param(est2.init_params_caiman, "fr") == get_init_param(
        est.init_params_caiman, "fr"
    )
    assert get_init_param(est2.init_params_caiman, "ar_order") == 2
    assert get_init_param(est2.init_params_caiman, "fudge_factor") == pytest.approx(0.97)


def test_ops_subgroups_roundtrip(saved_path):
    path, _, _, ops = saved_path
    _, _, ops2 = load_session(path)
    assert ops2.eval_method        == ops.eval_method
    assert ops2.contour_thr        == pytest.approx(ops.contour_thr)
    assert ops2.foopsi.fudge_factor == pytest.approx(ops.foopsi.fudge_factor)
    assert ops2.foopsi.ar_order    == ops.foopsi.ar_order
    assert ops2.foopsi.solver      == ops.foopsi.solver
    assert ops2.merge.method       == ops.merge.method
    assert ops2.merge.spatial_thr  == pytest.approx(ops.merge.spatial_thr)


def test_dims_and_indices_roundtrip(saved_path):
    path, est, _, _ = saved_path
    est2, _, _ = load_session(path)
    assert est2.dims == est.dims
    assert np.array_equal(est2.idx_components, est.idx_components)
    assert np.array_equal(est2.idx_components_bad, est.idx_components_bad)
    assert est2.num_cells_original == est.num_cells_original


def test_merge_parents_roundtrip(tiny_session, tmp_path):
    """merge_parents is persisted in session save so Reset Merges works
    across save/reload cycles. Empty list also round-trips correctly."""
    from caiman_sorter_py.io.session import save_session, load_session
    est, proc, ops = tiny_session
    proc.merge_parents = [(0, 1), (2, 3)]   # synthetic merge history
    p = tmp_path / "with_merges.h5"
    save_session(p, est, proc, ops)
    _, proc2, _ = load_session(p)
    assert proc2.merge_parents == [(0, 1), (2, 3)]


def test_merge_parents_empty_roundtrip(tiny_session, tmp_path):
    """Sessions without merges write no merge_parents dataset; reload picks
    up the dataclass default empty list."""
    from caiman_sorter_py.io.session import save_session, load_session
    est, proc, ops = tiny_session
    proc.merge_parents = []
    p = tmp_path / "no_merges.h5"
    save_session(p, est, proc, ops)
    _, proc2, _ = load_session(p)
    assert proc2.merge_parents == []
