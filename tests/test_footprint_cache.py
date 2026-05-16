"""Tests for core.footprint.FootprintCache — FIFO-bounded cache used by
image_panel and nav_panel.
"""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.footprint import FootprintCache


def test_get_returns_correct_shape_and_layout(tiny_session):
    """The cached footprint must be (h, w) and equal to the column-major
    reshape of est.A[:, idx]."""
    est, _, _ = tiny_session
    cache = FootprintCache(maxsize=10)
    fp = cache.get(est, 0)
    assert fp.shape == est.dims
    expected = est.A[:, 0].toarray().ravel().reshape(est.dims, order="F")
    np.testing.assert_array_equal(fp, expected)


def test_get_is_memoized(tiny_session):
    """A repeat call returns the same object, not a fresh array."""
    est, _, _ = tiny_session
    cache = FootprintCache(maxsize=10)
    fp1 = cache.get(est, 1)
    fp2 = cache.get(est, 1)
    assert fp1 is fp2
    assert 1 in cache


def test_fifo_eviction_at_cap(tiny_session):
    """When the cache is full, inserting a new cell drops the oldest."""
    est, _, _ = tiny_session
    cache = FootprintCache(maxsize=3)
    for i in range(3):
        cache.get(est, i)
    assert len(cache) == 3
    # Insert cell 3 → cell 0 should be evicted (oldest insertion).
    cache.get(est, 3)
    assert len(cache) == 3
    assert 0 not in cache
    assert 3 in cache and 1 in cache and 2 in cache


def test_clear_drops_all(tiny_session):
    est, _, _ = tiny_session
    cache = FootprintCache(maxsize=10)
    cache.get(est, 0); cache.get(est, 1); cache.get(est, 2)
    assert len(cache) == 3
    cache.clear()
    assert len(cache) == 0
    assert 0 not in cache


def test_lookup_miss_after_eviction_recomputes_correctly(tiny_session):
    """An evicted cell, when re-requested, returns the same value as the
    initial lookup (no stale state)."""
    est, _, _ = tiny_session
    cache = FootprintCache(maxsize=2)
    fp0_first = cache.get(est, 0).copy()
    cache.get(est, 1)
    cache.get(est, 2)   # evicts 0
    assert 0 not in cache
    fp0_again = cache.get(est, 0)
    np.testing.assert_array_equal(fp0_first, fp0_again)
