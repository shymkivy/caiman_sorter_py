"""Tests for the display-orientation transform helpers (core/orient.py)."""
from __future__ import annotations

import numpy as np
import pytest

from caiman_sorter_py.core.orient import (
    displayed_dims, inverse_xy, transform_image, transform_xy,
)
from caiman_sorter_py.core.state import PlotParams


# All 16 combinations of (rotation in 0/90/180/270) × flip_h × flip_v
ALL_ORIENTATIONS = [
    PlotParams(rotation=r, flip_h=fh, flip_v=fv)
    for r in (0, 90, 180, 270)
    for fh in (False, True)
    for fv in (False, True)
]


def _id(pp: PlotParams) -> str:
    return f"rot{pp.rotation}_fh{int(pp.flip_h)}_fv{int(pp.flip_v)}"


# --------------------------------------------------------------------------- #
# displayed_dims
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pp", ALL_ORIENTATIONS, ids=_id)
def test_displayed_dims_only_rotation_swaps_axes(pp):
    H, W = 5, 7
    out = displayed_dims((H, W), pp)
    if (pp.rotation // 90) % 2 == 0:
        assert out == (H, W)
    else:
        assert out == (W, H)


# --------------------------------------------------------------------------- #
# transform_image
# --------------------------------------------------------------------------- #

def test_transform_image_identity():
    arr = np.arange(15).reshape(3, 5)
    pp = PlotParams()
    out = transform_image(arr, pp)
    np.testing.assert_array_equal(out, arr)


def test_transform_image_rotation_matches_np_rot90():
    arr = np.arange(15).reshape(3, 5)
    for k, deg in [(1, 90), (2, 180), (3, 270)]:
        out = transform_image(arr, PlotParams(rotation=deg))
        np.testing.assert_array_equal(out, np.rot90(arr, k=k))


def test_transform_image_flips():
    arr = np.arange(15).reshape(3, 5)
    np.testing.assert_array_equal(
        transform_image(arr, PlotParams(flip_h=True)), arr[:, ::-1])
    np.testing.assert_array_equal(
        transform_image(arr, PlotParams(flip_v=True)), arr[::-1, :])
    np.testing.assert_array_equal(
        transform_image(arr, PlotParams(flip_h=True, flip_v=True)), arr[::-1, ::-1])


@pytest.mark.parametrize("pp", ALL_ORIENTATIONS, ids=_id)
def test_transform_image_sum_invariant(pp):
    """Flips and 90° rotations are isometries — pixel sum doesn't change."""
    rng = np.random.default_rng(42)
    arr = rng.normal(size=(8, 13))
    out = transform_image(arr, pp)
    assert out.sum() == pytest.approx(arr.sum())


@pytest.mark.parametrize("pp", ALL_ORIENTATIONS, ids=_id)
def test_transform_image_shape_matches_displayed_dims(pp):
    arr = np.arange(8 * 13).reshape(8, 13)
    out = transform_image(arr, pp)
    assert out.shape == displayed_dims(arr.shape, pp)


# --------------------------------------------------------------------------- #
# transform_xy / inverse_xy — round-trip exact for every orientation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pp", ALL_ORIENTATIONS, ids=_id)
def test_transform_xy_inverse_round_trip(pp):
    """transform_xy ∘ inverse_xy should be identity for every orientation."""
    H, W = 9, 11
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, W - 1, size=50)
    ys = rng.uniform(0, H - 1, size=50)
    xy = np.column_stack([xs, ys])
    disp, _ = transform_xy(xy, (H, W), pp)
    # invert each row back
    back = np.array([inverse_xy(disp[i, 0], disp[i, 1], (H, W), pp)
                     for i in range(len(disp))])
    np.testing.assert_allclose(back, xy, atol=1e-9)


@pytest.mark.parametrize("pp", ALL_ORIENTATIONS, ids=_id)
def test_inverse_xy_agrees_with_image_pixel_lookup(pp):
    """The pixel at (x_orig, y_orig) in arr must equal arr_disp[y_disp, x_disp]."""
    H, W = 7, 13
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(H, W))
    disp = transform_image(arr, pp)
    Hd, Wd = displayed_dims((H, W), pp)
    assert disp.shape == (Hd, Wd)

    # Sample every cell of the displayed image and verify the inverse maps
    # to the matching pixel in the original.
    for yd in range(Hd):
        for xd in range(Wd):
            x_orig, y_orig = inverse_xy(xd, yd, (H, W), pp)
            x_i = int(round(x_orig))
            y_i = int(round(y_orig))
            assert 0 <= x_i < W and 0 <= y_i < H
            assert arr[y_i, x_i] == pytest.approx(disp[yd, xd])


# --------------------------------------------------------------------------- #
# Empty / edge cases
# --------------------------------------------------------------------------- #

def test_transform_xy_handles_empty():
    out, dims = transform_xy(np.zeros((0, 2)), (5, 7), PlotParams(rotation=90))
    assert len(out) == 0
    assert dims == (7, 5)


def test_rotation_360_wraps_to_zero():
    """Non-canonical rotation values still wrap mod 360 — defensive parsing."""
    arr = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(
        transform_image(arr, PlotParams(rotation=360)), arr)
    np.testing.assert_array_equal(
        transform_image(arr, PlotParams(rotation=450)),
        np.rot90(arr, k=1))
