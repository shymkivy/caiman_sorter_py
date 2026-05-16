"""Headless tests for ui.image_panel invariants.

We test the cache-invalidation invariant in `refresh_images` without
spinning up a full Qt UI: pull the unbound method off `ImagePanel` and
invoke it on a plain stub object that has the attributes the method reads.
This way no QWidget / QApplication is required.
"""
from __future__ import annotations

import numpy as np
import pytest


class _DummySession:
    def __init__(self, est=None, proc=None):
        self.est = est
        self.proc = proc
        self.current_cell = 0


class _StubPanel:
    """Plain object exposing every attribute refresh_images touches.

    `refresh_images` is called as `ImagePanel.refresh_images(stub)` so Python
    never has to look up `__class__` for Qt parentage — we get the same code
    path with a tiny test double.
    """
    def __init__(self, est=None, proc=None):
        self.session            = _DummySession(est=est, proc=proc)
        self._A_csr             = None
        self._bkg_comp_weights  = None
        self._bkg_bg_img        = None
        self._acc_components    = None
        self._rej_components    = None
        self._acc_wcomp_signed  = None
        self._rej_wcomp_signed  = None
        self._footprint_cache   = {}

    # Stub out every helper refresh_images delegates to.
    def _ensure_bkg_cache(self):       pass
    def _clear_axes(self):             pass
    def _draw_backgrounds(self):       pass
    def _draw_all_contours(self):      pass
    def _update_labels(self):          pass
    def highlight_cell(self, _idx):    pass


def _refresh_images():
    """Pull the unbound method off ImagePanel without instantiating it."""
    from caiman_sorter_py.ui.image_panel import ImagePanel
    return ImagePanel.refresh_images


def test_refresh_images_no_session_is_noop():
    """With no est loaded, refresh_images must bail out before any work."""
    stub = _StubPanel()
    stub._A_csr = "sentinel"
    _refresh_images()(stub)
    assert stub._A_csr == "sentinel"


def test_refresh_images_clears_A_csr(tiny_session):
    """With a loaded session, refresh_images resets _A_csr to None so the
    next _on_click rebuilds against the (possibly post-merge) est.A.
    """
    est, proc, _ = tiny_session
    stub = _StubPanel(est=est, proc=proc)
    stub._A_csr = est.A.tocsr()    # simulate cached CSR from a prior click
    _refresh_images()(stub)
    assert stub._A_csr is None, (
        "_A_csr must be cleared by refresh_images so the next click "
        "rebuilds against post-merge est.A"
    )
