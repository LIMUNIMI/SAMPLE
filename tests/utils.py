"""Utilities for test cases"""
import copy
import itertools
import os
import unittest
from typing import Callable, Container, Iterable, Optional
import functools

import numpy as np
from chromatictools import unittestmixins

more_tests = unittest.skipUnless(os.environ.get("SAMPLE_MORE_TESTS", False),
                                 "enabled only if SAMPLE_MORE_TESTS is set")


def coherence_check(self: unittest.TestCase,
                    fwd: Callable,
                    bak: Callable,
                    modes: Optional[Iterable[str]] = None,
                    mode_key: str = "mode",
                    f=np.linspace(0, 2e4, 1024),
                    no_fwd: Optional[Container[str]] = None,
                    no_bak: Optional[Container[str]] = None,
                    **kwargs):
  """Check coherence of models

  Args:
    self (TestCase): Test case. Must also be an instance of
      :class:`chromatictools.unittestmixins.RMSEAssertMixin` and
      :class:`chromatictools.unittestmixins.AssertDoesntRaiseMixin`
    fwd (callable): Forward conversion function
    bak (callable): Backward conversion function
    modes (iterable of str): Modes to test
    f (array): Values to convert
    no_fwd (container of str): Modes with unsupported forward conversion
    no_bak (container of str): Modes with unsupported backward conversion
    kwargs: Keyword arguments for
      :func:`unittestmixins.RMSEAssertMixin.assert_almost_equal_rmse`"""
  self.assertIsInstance(self, unittestmixins.RMSEAssertMixin)
  self.assertIsInstance(self, unittestmixins.AssertDoesntRaiseMixin)
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


def coherence_check_method(method=None, **kwargs):
  if method is None:
    return functools.partial(coherence_check_method, **kwargs)

  @functools.wraps(method)
  def method_(self, *args, **kw):
    coherence_check(self, **kwargs)
    return method(self, *args, **kw)

  return method_
