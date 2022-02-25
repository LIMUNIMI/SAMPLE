"""Tests for psychoacoustic models"""
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
                      modes: Iterable[str],
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
    for m in modes:
      with self.subTest(mode=m, conversion="forward"):
        if m in no_fwd:
          with self.assertRaises(ValueError):
            fwd(f, mode=m)
        else:
          with self.assert_doesnt_raise():
            b = fwd(f, mode=m)
      with self.subTest(mode=m, conversion="backward"):
        if m in no_fwd:
          pass
        elif m in no_bak:
          with self.assertRaises(ValueError):
            bak(b, mode=m)
        else:
          f_ = bak(b, mode=m)
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
