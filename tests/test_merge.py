"""Tests for core.merge: duplicate detection + apply paths + idx sync."""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.merge import (
    _compute_merged_at,
    DuplicatePair,
    apply_choose_best_snr,
    apply_create_new,
    find_duplicate_pairs,
    reset_all_merges,
    sync_idx_components,
)


def test_find_duplicate_pairs_detects_overlapping_pair(tiny_session):
    """Cells 0 and 1 in the fixture overlap spatially AND share a temporal trace."""
    est, proc, ops = tiny_session
    # Use defaults — the fixture is built to exceed spatial_thr=0.5 and temporal_thr=0.5
    pairs = find_duplicate_pairs(est, proc, ops)
    assert len(pairs) >= 1
    pair_ids = {(p.cell_a, p.cell_b) for p in pairs}
    pair_ids_norm = {tuple(sorted(p)) for p in pair_ids}
    assert (0, 1) in pair_ids_norm, (
        f"expected the (0, 1) duplicate pair to be detected; got {pair_ids_norm}"
    )
    # The detected pair must report a sensible spatial overlap and temporal corr
    p01 = next(p for p in pairs
               if tuple(sorted((p.cell_a, p.cell_b))) == (0, 1))
    assert p01.spatial_overlap > ops.merge.spatial_thr
    assert p01.temporal_corr   > ops.merge.temporal_thr


def test_find_duplicate_pairs_respects_thresholds(tiny_session):
    """Raising spatial_thr above the overlap level should drop all pairs."""
    est, proc, ops = tiny_session
    ops.merge.spatial_thr = 1e9
    assert find_duplicate_pairs(est, proc, ops) == []


def test_find_duplicate_pairs_use_accepted_only(tiny_session):
    """Rejected cells are skipped when use_accepted_only=True."""
    est, proc, ops = tiny_session
    proc.accepted[1] = False   # take cell 1 out of the accepted pool
    ops.merge.use_accepted_only = True
    pairs = find_duplicate_pairs(est, proc, ops)
    ids = {tuple(sorted((p.cell_a, p.cell_b))) for p in pairs}
    assert (0, 1) not in ids


def test_apply_choose_best_snr_rejects_lower_snr(tiny_session):
    est, proc, ops = tiny_session
    # Make cell 0 the SNR winner over cell 1 (fixture already does this:
    # SNR2_vals[0]=10, [1]=8).
    pairs = find_duplicate_pairs(est, proc, ops)
    assert pairs
    result = apply_choose_best_snr(proc, pairs)
    assert result["n_changed"] >= 1
    # Loser (cell 1) must now be rejected; winner (cell 0) stays accepted.
    p = next(p for p in pairs if tuple(sorted((p.cell_a, p.cell_b))) == (0, 1))
    if proc.SNR2_vals[p.cell_a] >= proc.SNR2_vals[p.cell_b]:
        winner, loser = p.cell_a, p.cell_b
    else:
        winner, loser = p.cell_b, p.cell_a
    assert proc.accepted[winner] is np.True_ or proc.accepted[winner]
    assert not proc.accepted[loser]
    # The auto-eval state must reflect the manual reject as well so that a
    # subsequent reevaluate doesn't resurrect the loser.
    assert proc.accepted_core is None or not proc.accepted_core[loser]
    assert proc.manual_override[loser]


def test_sync_idx_components_after_reject(tiny_session):
    """sync_idx_components rebuilds idx_components/_bad from proc.accepted."""
    est, proc, ops = tiny_session
    # Flip a couple of cells without touching est.idx_components yet.
    proc.accepted = np.array([True, False, True, False, True])
    sync_idx_components(est, proc)
    assert list(est.idx_components)     == [0, 2, 4]
    assert list(est.idx_components_bad) == [1, 3]


def test_apply_create_new_appends_cell(tiny_session):
    """`weighted ave` should create one new accepted cell and reject both parents.

    Uses 'weighted ave' (cheapest non-trivial method) and the configured foopsi
    solver. The test only asserts structural invariants — not that foopsi
    produced any particular numerical result — so it stays robust to small
    solver differences.
    """
    pytest.importorskip("caiman")

    est, proc, ops = tiny_session
    ops.merge.method = "weighted ave"
    ops.foopsi.ar_order = 1   # fast path

    n0 = est.A.shape[1]
    pairs = find_duplicate_pairs(est, proc, ops)
    assert pairs, "fixture must yield at least one duplicate pair"

    result = apply_create_new(est, proc, ops, [pairs[0]])

    n_new = n0 + 1
    assert result["n_changed"] == 1
    assert est.A.shape[1] == n_new
    assert est.C.shape   == (n_new, proc.num_frames)
    assert est.YrA.shape == (n_new, proc.num_frames)
    assert est.S.shape   == (n_new, proc.num_frames)
    assert proc.accepted.shape == (n_new,)
    assert proc.accepted[-1], "new merged cell must be accepted"

    parents = result["rejected"][0]
    for p in parents:
        assert not proc.accepted[p], f"parent {p} must be rejected after merge"
        assert proc.manual_override[p]

    # est.idx_components is the pristine CaImAn-original record — merge must
    # NOT touch it. The new merged cell (at index n0) is correctly absent
    # from idx_components since it wasn't in CaImAn's auto-eval.
    assert n0 not in est.idx_components, (
        "merge must not pollute pristine est.idx_components with new merged cells"
    )
    # Live acceptance state lives in proc.accepted only.
    assert proc.accepted[n0], "new merged cell is accepted in proc.accepted"

    # Parents are recorded for "Reset merges" undo.
    assert len(proc.merge_parents) == 1
    pa, pb = proc.merge_parents[0]
    assert {pa, pb} == set(parents)


def test_reset_all_merges_restores_parents_and_truncates(tiny_session):
    """After apply_create_new + reset_all_merges:
       - merge_parents is empty
       - num_cells is back to original
       - parents have manual_override cleared and accepted = accepted_core
       - all per-cell arrays truncated to original length
    """
    pytest.importorskip("caiman")

    est, proc, ops = tiny_session
    ops.merge.method = "weighted ave"
    ops.foopsi.ar_order = 1

    n0 = est.A.shape[1]
    est.num_cells_original = n0   # explicit, since fixture leaves a default
    pairs = find_duplicate_pairs(est, proc, ops)
    parents = (pairs[0].cell_a, pairs[0].cell_b)
    # Pre-merge state of parents (the pristine auto-eval state)
    pre_core = proc.accepted_core[list(parents)].copy()

    apply_create_new(est, proc, ops, [pairs[0]])
    assert est.A.shape[1] == n0 + 1
    assert len(proc.merge_parents) == 1

    n_undone = reset_all_merges(est, proc)
    assert n_undone == 1
    assert len(proc.merge_parents) == 0
    assert proc.num_cells == n0
    assert est.A.shape[1] == n0
    assert est.C.shape   == (n0, proc.num_frames)
    assert est.YrA.shape == (n0, proc.num_frames)
    assert est.S.shape   == (n0, proc.num_frames)
    assert proc.accepted.shape        == (n0,)
    assert proc.accepted_core.shape   == (n0,)
    assert proc.manual_override.shape == (n0,)
    for p, expected in zip(parents, pre_core):
        assert proc.accepted[p] == bool(expected)
        assert not proc.manual_override[p]


def test_reset_all_merges_noop_when_no_merges(tiny_session):
    est, proc, _ = tiny_session
    proc.merge_parents = []   # ensure clean
    n_before = proc.num_cells
    n_undone = reset_all_merges(est, proc)
    assert n_undone == 0
    assert proc.num_cells == n_before


def test_weighted_ave_baseline_uses_min_C_only(tiny_session):
    """`weighted ave` merge weights A by ||trace - base|| where the baseline
    is min(C) of each parent (MATLAB f_cs_find_similar_comp_core.m:28),
    NOT min(C + YrA). Verified end-to-end via _compute_merged_at.
    """
    est, _, _ = tiny_session
    # Inject a controlled trace where min(C) ≠ min(C + YrA) so the two
    # baseline conventions yield distinguishable A_new values.
    a, b = 0, 1
    n_frames = est.C.shape[1]
    rng = np.random.default_rng(123)
    # Cell a: C has a deep dip; YrA has noise around 0.
    est.C[a]   = rng.standard_normal(n_frames) * 0.5
    est.C[a, 100] = -3.0                                 # min(C) = -3
    est.YrA[a] = rng.standard_normal(n_frames) * 0.3
    est.YrA[a, 200] = 1.5                                # doesn't lower min
    # Cell b: typical
    est.C[b]   = rng.standard_normal(n_frames) * 0.5
    est.YrA[b] = rng.standard_normal(n_frames) * 0.3

    pair = DuplicatePair(cell_a=a, cell_b=b,
                         spatial_overlap=1.0, temporal_corr=1.0)
    A_new, _ = _compute_merged_at(est, pair, "weighted ave")

    # Expected A_new uses base = min(est.C[i]).
    A1 = est.A[:, a].toarray().ravel()
    A2 = est.A[:, b].toarray().ravel()
    tr1 = est.C[a] + est.YrA[a]
    tr2 = est.C[b] + est.YrA[b]
    base1 = float(est.C[a].min())
    base2 = float(est.C[b].min())
    n1 = float(np.linalg.norm(tr1 - base1))
    n2 = float(np.linalg.norm(tr2 - base2))
    A_expected = (A1 * n1 + A2 * n2) / (n1 + n2)
    np.testing.assert_allclose(A_new, A_expected, rtol=1e-10)

    # Also assert it DIFFERS from the wrong (min of C+YrA) convention to
    # guarantee the test would catch a regression to that code.
    n1_wrong = float(np.linalg.norm(tr1 - tr1.min()))
    n2_wrong = float(np.linalg.norm(tr2 - tr2.min()))
    A_wrong = (A1 * n1_wrong + A2 * n2_wrong) / (n1_wrong + n2_wrong)
    assert not np.allclose(A_new, A_wrong, rtol=1e-6), (
        "test setup must produce distinguishable A_new under the two baseline "
        "conventions; check C/YrA injection"
    )
