"""Tests for the overall method"""
import copy
import itertools
import json
import unittest

import numpy as np
import sample
import sample.sample
from chromatictools import unittestmixins
from sample import plots


class TestSAMPLE(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Tests for the overall method"""

  def setUp(self) -> None:
    """Initialize test audio and SAMPLE model"""
    self.fs = 44100
    self.x = sample.sample.additive_synth(np.arange(int(2 * self.fs)) / self.fs,
      freqs=np.array([440, 650, 690]),
      amps=np.array([1, .5, .45]),
      decays=np.array([.66, .4, .35]),
    )
    np.random.seed(42)
    self.noise = np.random.randn(*self.x.shape)
    self.x += self.noise * np.power(10, -60 / 20)
    self.x /= np.max(np.abs(self.x))
    self.sample = sample.SAMPLE(
        sinusoidal_model__fs=self.fs,
        sinusoidal_model__max_n_sines=10,
        sinusoidal_model__peak_threshold=-30,
    )
    stc = self.sample.get_params()["sinusoidal_model__sine_tracker_cls"]
    stk = self.sample.sinusoidal_model.sine_tracker_kwargs
    self.st = stc(**stk)

  def test_no_exceptions(self):
    """Test that no exceptions arise from method"""
    s = copy.deepcopy(self.sample)
    with self.assert_doesnt_raise():
      s.fit(self.x).predict(np.arange(self.x.size) / self.fs)

  def test_no_exceptions_reverse(self):
    """Test that no exceptions arise from method using reverse mode"""
    s = copy.deepcopy(self.sample).set_params(sinusoidal_model__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x).predict(np.arange(self.x.size) / self.fs)

  def test_no_exceptions_less_modes(self):
    """Test that no exceptions arise from method
    using a reduced number of modes"""
    s = copy.deepcopy(self.sample).set_params(
        sinusoidal_model__reverse=True,
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

  def test_dur_too_short(self):
    """Test that errors occur for too short sine durations"""
    s = copy.deepcopy(self.sample).set_params(
        sinusoidal_model__min_sine_dur=1e-6,)
    with self.assertRaises(ValueError):
      s.fit(self.noise)

  def test_no_exceptions_safe_len(self):
    """Test no errors occur with safe length parameter"""
    s = copy.deepcopy(self.sample).set_params(
        sinusoidal_model__min_sine_dur=1e-6,
        sinusoidal_model__safe_sine_len=2,
    )
    with self.assert_doesnt_raise():
      s.fit(self.noise)

  def test_tracker_reset(self):
    """Test tracker reset"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.subTest(step="chek state is non-empty", var="tracks"):
      self.assertNotEqual(len(s.sinusoidal_model.sine_tracker_.tracks_), 0)  # pylint: disable=W0212
    with self.subTest(step="chek state is non-empty", var="_active_tracks"):
      self.assertNotEqual(
          len(s.sinusoidal_model.sine_tracker_._active_tracks),  # pylint: disable=W0212
          0)
    with self.subTest(step="chek state is non-empty", var="_frame"):
      self.assertNotEqual(s.sinusoidal_model.sine_tracker_._frame, 0)  # pylint: disable=W0212
    s.sinusoidal_model.sine_tracker_.reset()
    with self.subTest(step="chek state is reset", var="tracks"):
      self.assertEqual(len(s.sinusoidal_model.sine_tracker_.tracks_), 0)  # pylint: disable=W0212
    with self.subTest(step="chek state is reset", var="_active_tracks"):
      self.assertEqual(len(s.sinusoidal_model.sine_tracker_._active_tracks), 0)  # pylint: disable=W0212
    with self.subTest(step="chek state is reset", var="_frame"):
      self.assertEqual(s.sinusoidal_model.sine_tracker_._frame, 0)  # pylint: disable=W0212

  def test_refit_deletes_intermediate(self):
    """Test intermediate results reset"""
    s = copy.deepcopy(self.sample).set_params(
        sinusoidal_model__save_intermediate=True).fit(self.x)
    with self.subTest(step="interediate_saved"):
      self.assertTrue(hasattr(s.sinusoidal_model, "intermediate_"))
    s.set_params(sinusoidal_model__save_intermediate=False).fit(self.x)
    with self.subTest(step="interediate_reset"):
      self.assertFalse(hasattr(s.sinusoidal_model, "intermediate_"))

  def test_merge_strategy_raises(self):
    """Test that an unsupported merging strategy causes an Exception"""
    s = copy.deepcopy(self.sample)
    s.set_params(
        sinusoidal_model__merge_strategy="unsupported merging strategy")
    with self.assertRaises(KeyError):
      s.fit(self.x)

  def test_single_merge_strategy(self):
    """Test :data:`"single"` merging strategy"""
    s = copy.deepcopy(self.sample)
    s.set_params(sinusoidal_model__merge_strategy="single")
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_strip(self):
    """Test stripping tracks"""
    s = copy.deepcopy(self.sample)
    s.set_params(sinusoidal_model__strip_t=0.01)
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_strip_reverse(self):
    """Test stripping tracks in reverse mode"""
    s = copy.deepcopy(self.sample)
    s.set_params(sinusoidal_model__strip_t=0.01, sinusoidal_model__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_freqs_modification(self):
    """Test that frequencies can be effectively manipulated post-fit"""
    s = copy.deepcopy(self.sample)
    s.fit(self.x)
    t = copy.deepcopy(s)
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
      kw = dict(sinusoidal_model__reverse=r,
                sinusoidal_model__save_intermediate=s)
      with self.subTest(**kw):
        m = copy.deepcopy(self.sample)
        m.set_params(**kw)
        m.fit(self.x)
        with self.assert_doesnt_raise():
          plots.sine_tracking_2d(m.sinusoidal_model)
          plots.plt.clf()

  def test_plot_3d(self):
    """Test 3D plot"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.assert_doesnt_raise():
      plots.sine_tracking_3d(s.sinusoidal_model)
