"""Tests for optimzation routines"""
import copy
import itertools
import unittest

import skopt.space
from chromatictools import unittestmixins

import sample
from sample import optimize
from sample.evaluation import random


class TestRemapper(unittest.TestCase):
  """Tests for argument remapper"""

  def test_passthrough(self):
    """Test that provided targets pass through"""
    kwargs = dict(
        sinusoidal__n=128,
        sinusoidal__w=list(itertools.repeat(1 / 128, 128)),
        sinusoidal__h=64,
    )
    self.assertEqual(kwargs, optimize.sample_kwargs_remapper(**kwargs))

  def test_wsize(self,
                 fft_log_ns=(7, 8, 11),
                 w_sizes=(0.25, 0.5, 0.75),
                 w_types=("hamming", "blackman")):
    """Test that window size is correct"""
    for log_n, ws, wt in itertools.product(fft_log_ns, w_sizes, w_types):
      kwargs = optimize.sample_kwargs_remapper(
          sinusoidal__log_n=log_n,
          sinusoidal__wsize=ws,
          sinusoidal__wtype=wt,
      )
      with self.subTest(log_n=log_n, ws=ws, wt=wt):
        self.assertEqual(kwargs["sinusoidal__w"].size, int((1 << log_n) * ws))

  def test_hsize(
      self,
      fft_log_ns=(7, 8, 11),
      w_sizes=(0.25, 0.5, 0.75),
      olaps=(0.25, 0.5, 0.75),
  ):
    """Test that hop size is correct"""
    for log_n, ws, olap in itertools.product(fft_log_ns, w_sizes, olaps):
      kwargs = optimize.sample_kwargs_remapper(
          sinusoidal__log_n=log_n,
          sinusoidal__wsize=ws,
          sinusoidal__overlap=olap,
      )
      with self.subTest(log_n=log_n, ws=ws, olap=olap):
        self.assertAlmostEqual(
            olap, 1 - kwargs["sinusoidal__h"] / kwargs["sinusoidal__w"].size)


class TestOptimize(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Tests for optimizer class"""

  def setUp(self) -> None:
    """Initialize test audio and SAMPLE model"""
    self.x, self.fs, _ = random.BeatsGenerator(seed=1234).audio()
    self.sample_opt = optimize.SAMPLEOptimizer(
        model=sample.SAMPLE(
            max_n_modes=3,
            sinusoidal__tracker__reverse=True,
            sinusoidal__tracker__frequency_bounds=(50, 20e3),
        ),
        sinusoidal__log_n=skopt.space.Integer(6, 15, name="log2(n)"),
        sinusoidal__tracker__max_n_sines=skopt.space.Integer(16,
                                                             128,
                                                             name="n sines"),
        sinusoidal__tracker__peak_threshold=skopt.space.Real(
            -120, -30, name="peak threshold"),
        sinusoidal__tracker__min_sine_dur=skopt.space.Integer(
            2, 42, name="min duration"),
        sinusoidal__overlap=skopt.space.Real(0, 0.75, name="overlap"),
    )

  def test_no_exceptions(self):
    """Test that no exceptions arise during optimization"""
    with self.assert_doesnt_raise():
      copy.deepcopy(self.sample_opt).gp_minimize(
          x=self.x,
          fs=self.fs,
          n_calls=8,
          n_initial_points=4,
      )

  def test_no_exceptions_tqdm(self, n_calls=8, n_initial_points=4):
    """Test tqdm callback"""
    sample_opt = copy.deepcopy(self.sample_opt)
    callback = optimize.TqdmCallback(
        sample_opt=sample_opt,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
    )
    with self.assert_doesnt_raise():
      sample_opt.gp_minimize(x=self.x,
                             fs=self.fs,
                             n_calls=n_calls,
                             n_initial_points=n_initial_points,
                             callback=callback)
    with self.subTest(test="i == n_calls"):
      self.assertEqual(callback.i, n_calls)
    callback.reset().start()
    with self.subTest(test="reset"):
      self.assertEqual(callback.i, 0)

  def test_optimize_continue(self, n_calls=4, n_initial_points=2, times=2):
    """Test continuing optimization"""
    sample_opt = copy.deepcopy(self.sample_opt)
    res = None
    for i in range(1, times + 1):
      with self.assert_doesnt_raise():
        _, res = sample_opt.gp_minimize(x=self.x,
                                        fs=self.fs,
                                        n_calls=n_calls,
                                        n_initial_points=n_initial_points,
                                        state=res)
      with self.subTest(step=i):
        self.assertEqual(len(res.x_iters), i * n_calls)

  def test_error_on_kwargs(self):
    """Test error for incorrect number of arguments"""
    with self.assertRaises(ValueError):
      self.sample_opt.loss(self.x, self.fs)()
