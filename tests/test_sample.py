"""Tests for the overall method"""
import copy
import itertools
import json
import unittest

import numpy as np
from chromatictools import unittestmixins
from matplotlib import pyplot as plt
from sklearn import base

import sample
import sample.sample
from sample import plots
from sample.utils import dsp as dsp_utils


class TestSAMPLE(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Tests for the overall method"""

  def setUp(self) -> None:
    """Initialize test audio and SAMPLE model"""
    self.fs = 44100
    self.x = sample.sample.additive_synth(
        np.arange(int(2 * self.fs)) / self.fs,
        freqs=np.array([440, 650, 690]),
        amps=np.array([1, .5, .45]),
        decays=np.array([.66, .4, .35]),
    )
    np.random.seed(42)
    self.noise = np.random.randn(*self.x.shape)
    self.x += self.noise * dsp_utils.db2a(-60)
    self.x /= np.max(np.abs(self.x))
    self.sample = sample.SAMPLE(
        sinusoidal__tracker__fs=self.fs,
        sinusoidal__tracker__max_n_sines=10,
        sinusoidal__tracker__peak_threshold=-30,
    )
    self.st = self.sample.sinusoidal.tracker

  def test_no_exceptions(self):
    """Test that no exceptions arise from method"""
    s = base.clone(self.sample)
    with self.assert_doesnt_raise():
      s.fit(self.x)
      y = s.predict(np.arange(self.x.size) / self.fs)
    self.assertAlmostEqual(np.abs(y).max(), s.amps_.sum())

  def test_no_exceptions_reverse(self):
    """Test that no exceptions arise from method using reverse mode"""
    s = base.clone(self.sample).set_params(sinusoidal__tracker__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x)
      y = s.predict(np.arange(self.x.size) / self.fs)
    self.assertAlmostEqual(np.abs(y).max(), s.amps_.sum())

  def test_no_exceptions_random_phase(self):
    """Test random phase for synthesis"""
    s = base.clone(self.sample).set_params(sinusoidal__tracker__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x)
      y = s.predict(np.arange(self.x.size) / self.fs, phases="random")
    self.assertLessEqual(np.abs(y).max(), s.amps_.sum())

  def test_unsupported_option_phase(self):
    """Test exception raising on unsupported option for phase"""
    s = base.clone(self.sample).set_params(sinusoidal__tracker__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x)
    with self.assertRaises(ValueError):
      s.predict(np.arange(self.x.size) / self.fs, phases="unsupported")

  def test_no_exceptions_less_modes(self):
    """Test that no exceptions arise from method
    using a reduced number of modes"""
    s = base.clone(self.sample).set_params(
        sinusoidal__tracker__reverse=True,
        max_n_modes=4,
    )
    with self.assert_doesnt_raise():
      s.fit(self.x).predict(np.arange(self.x.size) / self.fs)

  def test_sdt_json_serializable(self):
    """Test that SDT parameters are JSON-serializable"""
    with self.assert_doesnt_raise():
      json.dumps(self.sample.fit(self.x).sdt_params_())

  def test_freq_too_high(self):
    """Test track rejection for high frequencies"""
    self.assertFalse(self.st.track_ok(dict(freq=np.full(1024, 1e9),)))

  def test_freq_too_low(self):
    """Test track rejection for low frequencies"""
    self.assertFalse(self.st.track_ok(dict(freq=np.zeros(1024),)))

  def test_tracker_reset(self):
    """Test tracker reset"""
    s = base.clone(self.sample).fit(self.x)
    with self.subTest(step="chek state is non-empty", var="tracks"):
      self.assertNotEqual(len(s.sinusoidal.tracker.tracks_), 0)
    with self.subTest(step="chek state is non-empty", var="_active_tracks"):
      self.assertNotEqual(
          len(s.sinusoidal.tracker._active_tracks),  # pylint: disable=W0212
          0)
    with self.subTest(step="chek state is non-empty", var="_frame"):
      self.assertNotEqual(s.sinusoidal.tracker._frame, 0)  # pylint: disable=W0212
    s.sinusoidal.tracker.reset()
    with self.subTest(step="chek state is reset", var="tracks"):
      self.assertEqual(len(s.sinusoidal.tracker.tracks_), 0)
    with self.subTest(step="chek state is reset", var="_active_tracks"):
      self.assertEqual(len(s.sinusoidal.tracker._active_tracks), 0)  # pylint: disable=W0212
    with self.subTest(step="chek state is reset", var="_frame"):
      self.assertEqual(s.sinusoidal.tracker._frame, 0)  # pylint: disable=W0212

  def test_refit_deletes_intermediate(self):
    """Test intermediate results reset"""
    s = base.clone(
        self.sample).set_params(sinusoidal__intermediate__save=True).fit(self.x)
    with self.subTest(step="interediate_saved"):
      self.assertTrue(hasattr(s.sinusoidal.intermediate, "cache_"))
    s.set_params(sinusoidal__intermediate__save=False).fit(self.x)
    with self.subTest(step="interediate_reset"):
      self.assertFalse(hasattr(s.sinusoidal.intermediate, "cache_"))

  def test_merge_strategy_raises(self):
    """Test that an unsupported merging strategy causes an Exception"""
    s = base.clone(self.sample)
    s.set_params(
        sinusoidal__tracker__merge_strategy="unsupported merging strategy")
    with self.assertRaises(KeyError):
      s.fit(self.x)

  def test_single_merge_strategy(self):
    """Test :data:`"single"` merging strategy"""
    s = base.clone(self.sample)
    s.set_params(sinusoidal__tracker__merge_strategy="single")
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_strip(self):
    """Test stripping tracks"""
    s = base.clone(self.sample)
    s.set_params(sinusoidal__tracker__strip_t=0.01)
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_strip_reverse(self):
    """Test stripping tracks in reverse mode"""
    s = base.clone(self.sample)
    s.set_params(sinusoidal__tracker__strip_t=0.01,
                 sinusoidal__tracker__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_freqs_modification(self):
    """Test that frequencies can be effectively manipulated post-fit"""
    s = base.clone(self.sample).fit(self.x)
    t = base.clone(s).fit(self.x)
    k = 1.2
    t.freqs_ *= k
    for i, (f_s, f_t) in enumerate(zip(s.freqs_, t.freqs_)):
      with self.subTest(partial=i):
        self.assertEqual(f_s * k, f_t)

  def test_decays_modification(self):
    """Test that decays can be effectively manipulated post-fit"""
    s = copy.deepcopy(self.sample)
    s.fit(self.x)
    t = copy.deepcopy(s)
    k = 1.2
    t.decays_ *= k
    for i, (d_s, d_t) in enumerate(zip(s.decays_, t.decays_)):
      with self.subTest(partial=i):
        self.assertEqual(d_s * k, d_t)

  def test_amps_modification(self):
    """Test that amplitudes can be effectively manipulated post-fit"""
    s = copy.deepcopy(self.sample)
    s.fit(self.x)
    t = copy.deepcopy(s)
    k = 1.2
    t.amps_ *= k
    for i, (a_s, a_t) in enumerate(zip(s.amps_, t.amps_)):
      with self.subTest(partial=i):
        self.assertEqual(a_s * k, a_t)

  def test_plot_2d(self):
    """Test 2D plot"""
    for r, s in itertools.product(*itertools.tee(map(bool, range(2)), 2)):
      kw = dict(sinusoidal__tracker__reverse=r,
                sinusoidal__intermediate__save=s)
      with self.subTest(**kw):
        m = copy.deepcopy(self.sample)
        m.set_params(**kw)
        m.fit(self.x)
        with self.assert_doesnt_raise():
          plots.sine_tracking_2d(m)
          plots.plt.clf()

  def test_plot_3d(self):
    """Test 3D plot"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.assert_doesnt_raise():
      plots.sine_tracking_3d(s)

  def test_plot_resynthesis(self):
    """Test resynthesis plot"""
    s = copy.deepcopy(self.sample).fit(self.x)
    plots.resynthesis(
        self.x, {"Resynthesis": s},
        wav_kws=dict(alpha=0.66),
        tf_kws=dict(cmap="inferno"),
        foreach=lambda i, k, y:
        (self.assertIsInstance(i, int), self.assertIsInstance(k, str),
         self.assertIsInstance(y, np.ndarray)))

  def test_plot_resynthesis_axs(self):
    """Test resynthesis plot specifying axes"""
    _, axs = plt.subplots(2, 1)
    with self.assert_doesnt_raise():
      plots.resynthesis(self.x, axs=axs)

  def test_parallel_fit(self):
    """Test SAMPLE fit in multiprocessing (useless)"""
    s = base.clone(self.sample).fit(self.x)
    p = base.clone(self.sample).fit(self.x, n_jobs=4)
    with self.subTest(test="freqs"):
      np.testing.assert_array_equal(s.freqs_, p.freqs_)
    with self.subTest(test="amps"):
      np.testing.assert_array_equal(s.amps_, p.amps_)
    with self.subTest(test="decays"):
      np.testing.assert_array_equal(s.decays_, p.decays_)

  def test_fit_no_tracks(self):
    """Test that SAMPLE doesn't find anything in low-volume noise"""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(scale=dsp_utils.db2a(-20), size=self.x.size)
    s = base.clone(self.sample).fit(x)
    with self.subTest(check="tracks"):
      self.assertEqual(s.sinusoidal.tracks_, [])
    with self.subTest(check="params"):
      np.testing.assert_array_equal(s.param_matrix_.flatten(), [])
    with self.subTest(check="audio"):
      np.testing.assert_array_equal(s.predict(np.arange(self.x.size) / self.fs),
                                    np.zeros(self.x.size))
