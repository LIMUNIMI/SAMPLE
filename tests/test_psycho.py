"""Tests for psychoacoustic models"""
import copy
import itertools
import unittest
from typing import Callable, Container, Iterable, Optional

import numpy as np
from chromatictools import unittestmixins
from sample import psycho


class TestPsycho(unittestmixins.AssertDoesntRaiseMixin,
                 unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests psychoacoustic models functions"""

  def _t3st_coherence(self,
                      fwd: Callable,
                      bak: Callable,
                      modes: Optional[Iterable[str]] = None,
                      mode_key: str = "mode",
                      f=np.linspace(0, 2e4, 1024),
                      no_fwd: Optional[Container[str]] = None,
                      no_bak: Optional[Container[str]] = None,
                      **kwargs):
    """Test coherence of models

    Args:
      fwd (callable): Forward conversion function
      bak (callable): Backward conversion function
      modes (iterable of str): Modes to test
      f: Frequency values (in Hertz) to convert
      no_fwd (container of str): Modes with unsupported forward conversion
      no_bak (container of str): Modes with unsupported backward conversion
      kwargs: Keyword arguments for
        :func:`unittestmixins.RMSEAssertMixin.assert_almost_equal_rmse`"""
    if no_fwd is None:
      no_fwd = ()
    if no_bak is None:
      no_bak = ()
    if modes is None:
      modes = [None]
    for m, use_buf in itertools.product(modes, (False, True)):
      kw = {} if m is None else {mode_key: m}
      subt_kw = copy.deepcopy(kw)
      if use_buf:
        kw["out"] = np.empty_like(f)
      subt_kw["buffer"] = use_buf
      with self.subTest(conversion="forward", **subt_kw):
        if m in no_fwd:
          with self.assertRaises(ValueError):
            fwd(f, **kw)
        else:
          with self.assert_doesnt_raise():
            b = fwd(f, **kw)
      with self.subTest(conversion="backward", **subt_kw):
        if m in no_fwd:
          pass
        elif m in no_bak:
          with self.assertRaises(ValueError):
            bak(b, **kw)
        else:
          f_ = bak(b, **kw)
          self.assert_almost_equal_rmse(f, f_, **kwargs)

  def test_bark(self):
    """Test coherence of conversion for Barks"""
    self._t3st_coherence(
        fwd=psycho.hz2bark,
        bak=psycho.bark2hz,
        modes=("unsupported", "zwicker", "traunmuller", "wang"),
        no_fwd=("unsupported",),
        no_bak=("zwicker",),
    )

  def test_mel(self):
    """Test coherence of conversion for Mels"""
    self._t3st_coherence(fwd=psycho.hz2mel,
                         bak=psycho.mel2hz,
                         modes=("unsupported", "default", "fant"),
                         no_fwd=("unsupported",))

  def test_db(self):
    """Test coherence of conversion for dB"""
    self._t3st_coherence(fwd=psycho.db2a,
                         bak=psycho.a2db,
                         f=np.linspace(-60, 60, 1024))

  def test_complex_db(self):
    """Test coherence of conversion for dB from complex"""
    self._t3st_coherence(fwd=lambda *args, **kwargs: psycho.db2a(
        *args, **kwargs).astype(complex),
                         bak=psycho.complex2db,
                         f=np.linspace(-60, 60, 1024))

  def test_db_floor(self):
    """Test floor for dB conversion"""
    f = -60
    a = np.linspace(0, 1, 1024)
    psycho.a2db(a, floor=f, floor_db=True, out=a)
    with self.subTest(check="dB"):
      self.assertTrue(np.greater_equal(a, f).all())
    psycho.db2a(a, out=a)
    with self.subTest(check="a"):
      self.assertTrue(np.greater_equal(a, psycho.db2a(f)).all())

  def test_cams(self):
    """Test coherence of conversion for ERB-rate scale"""
    self._t3st_coherence(fwd=psycho.hz2cams,
                         bak=psycho.cams2hz,
                         mode_key="degree",
                         modes=("unsupported", "linear", "quadratic"),
                         no_fwd=("unsupported",))

  def test_erb_monotonicity(self):
    """Test monotonicity of ERBs"""
    f = np.linspace(0, 2e4, 1024)
    for deg, use_buf in itertools.product(("linear", "quadratic"),
                                          (False, True)):
      kw = {} if use_buf else dict(out=np.empty_like(f))
      kw["degree"] = deg
      erbs = psycho.erb(f, **kw)
      with self.subTest(degree=deg, buffer=use_buf):
        self.assertTrue(np.greater(np.diff(erbs), 0).all())
