"""Tests for core.state.get_init_param.

The function has to cope with three on-disk shapes:
  1. Nested dict — CaImAn HDF5 stores params under data/init/temporal/preprocess/...
  2. Flat dict   — legacy session files and the MATLAB pipeline use top-level keys
  3. 1-element array / list — HDF5 / .mat round-trips wrap scalars in arrays.
"""
from __future__ import annotations

import numpy as np

from caiman_sorter_py.core.state import get_init_param


def test_returns_default_for_none_or_empty():
    assert get_init_param(None, "fr", 30.0) == 30.0
    assert get_init_param({}, "fr", 30.0) == 30.0


def test_returns_default_for_unknown_key():
    assert get_init_param({"fr": 60}, "totally_unknown_key", "x") == "x"


def test_flat_dict_lookup():
    init = {"fr": 60.0, "p": 2, "fudge_factor": 0.98}
    assert get_init_param(init, "fr") == 60.0
    assert get_init_param(init, "ar_order", default=999) == 999  # ar_order not in flat
    assert get_init_param(init, "fudge_factor") == 0.98


def test_nested_dict_lookup():
    init = {
        "data":       {"fr": 30.0},
        "preprocess": {"p": 2},
        "temporal":   {"fudge_factor": 0.97, "lags": 5},
        "init":       {"gSig": [3, 3]},
    }
    assert get_init_param(init, "fr") == 30.0
    assert get_init_param(init, "ar_order") == 2
    assert get_init_param(init, "fudge_factor") == 0.97
    assert get_init_param(init, "lags") == 5
    assert get_init_param(init, "gSig") == [3, 3]


def test_unwraps_single_element_numpy_array():
    init = {"fr": np.array([60.0])}
    out = get_init_param(init, "fr")
    assert out == 60.0
    assert not isinstance(out, np.ndarray)
    # int(...) should work without type errors
    assert int(out) == 60


def test_unwraps_single_element_list():
    init = {"fr": [60.0]}
    assert get_init_param(init, "fr") == 60.0


def test_unwraps_nested_single_element_array():
    init = {"data": {"fr": np.array([[30.0]])}}
    out = get_init_param(init, "fr")
    assert float(out) == 30.0


def test_does_not_unwrap_multi_element_array():
    init = {"data": {"fr": np.array([30.0, 60.0])}}
    out = get_init_param(init, "fr")
    assert isinstance(out, np.ndarray) and out.size == 2


def test_nested_then_flat_priority():
    """If a flat top-level key is present, it should be used regardless of nesting."""
    init = {
        "fr":   45.0,
        "data": {"fr": 30.0},
    }
    assert get_init_param(init, "fr") == 45.0


def test_missing_nested_branch_returns_default():
    """If part of the nested path is missing, return the default rather than crashing."""
    init = {"data": {}}
    assert get_init_param(init, "fr", default=12.5) == 12.5
