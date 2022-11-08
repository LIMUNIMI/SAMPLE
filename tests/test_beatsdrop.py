"""Test Beat models"""
import itertools
import unittest

import numpy as np
import sample
import sample.beatsdrop.regression
from chromatictools import unittestmixins
from sample import beatsdrop


class TestOther(unittest.TestCase):
  """Test other functions in the module"""

  def test_get_notnone(self):
    """Test _get_notnone_attr"""

    class C:
      """Class for testing _get_notnone_attr"""
      a = None
      b = None
      c = "c"

    c = C()

    with self.subTest(test="single option"):
      self.assertEqual(beatsdrop.regression._get_notnone_attr(c, "c"), "c")  # pylint: disable=W0212
    with self.subTest(test="multiple options"):
      self.assertEqual(
          beatsdrop.regression._get_notnone_attr(c, "a", "b", "c"),  # pylint: disable=W0212
          "c")
    with self.subTest(test="multiple invalid options"):
      self.assertEqual(
          beatsdrop.regression._get_notnone_attr(  # pylint: disable=W0212
              c, "undefined", "missing", "c"),
          "c")
    with self.subTest(test="only invalid options"):
      with self.assertRaises(AttributeError):
        beatsdrop.regression._get_notnone_attr(c, "undefined", "missing")  # pylint: disable=W0212


class TestBeat(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Test beat models"""

  def setUp(self):
    self.fs = 44100
    self.t = np.arange(self.fs) / self.fs

  def test_compute_everything(self):
    """Test that all node variables can be computed"""
    b = beatsdrop.Beat()
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)

  def test_compute_everything_modal(self):
    """Test that all node variables can be computed (modal beat)"""
    b = beatsdrop.ModalBeat()
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)

  def test_variable_error(self):
    """Test error on undefined variable"""
    b = beatsdrop.ModalBeat()
    with self.assertRaises(AttributeError):
      b.yourself(self.t)

  def test_argument_error(self):
    """Test error on time argument undefined"""
    b = beatsdrop.ModalBeat()
    errs = ("a0", "a1", "a_hat", "a_oln", "a_oln2", "a_hat2", "alpha2", "am",
            "fm", "pm", "x")
    for k in b.variables:
      e = k in errs
      with self.subTest(variable=k, error=e):
        if e:
          with self.assertRaises(ValueError):
            getattr(b, k)(None)
        else:
          with self.assert_doesnt_raise():
            getattr(b, k)(None)

  def test_lambda_ok(self):
    """Test that lambdas can be used"""
    b = beatsdrop.Beat(a0=lambda t: 1 / (1 + t))
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)


class TestBeatRegression(unittestmixins.RMSEAssertMixin,
                         unittestmixins.SignificantPlacesAssertMixin,
                         unittest.TestCase):
  """Test beat regression models"""

  def setUp(self):
    """Setup test signal and sinusoidal tracks"""
    self.a0, self.a1 = 0.8, 0.2
    f, am_f = 1100, np.pi
    self.f0, self.f1 = f + am_f, f - am_f
    self.d0, self.d1 = 0.75, 2
    np.random.seed(42)
    self.p0, self.p1 = np.random.rand(2) * 2 * np.pi
    self.fs = 44100
    self.t = np.arange(int(self.fs * 3)) / self.fs
    self.a0, self.a1, self.f0, self.f1, \
    self.d0, self.d1, self.p0, self.p1 = beatsdrop.regression.sort_params(
        (self.a0, self.a1, self.f0, self.f1,
         self.d0, self.d1, self.p0, self.p1))
    self.beat = beatsdrop.ModalBeat(
        a0=self.a0,
        a1=self.a1,
        f0=self.f0,
        f1=self.f1,
        d0=self.d0,
        d1=self.d1,
        p0=self.p0,
        p1=self.p1,
    )
    self.x, self.am, self.fm = self.beat.compute(self.t, ("x", "am", "fm"))
    self.fm /= 2 * np.pi
    model = sample.SAMPLE(
        sinusoidal_model__max_n_sines=32,
        sinusoidal_model__reverse=True,
        sinusoidal_model__t=-90,
        sinusoidal_model__save_intermediate=True,
        sinusoidal_model__peak_threshold=-45,
        sinusoidal_model__safe_sine_len=2,
    ).fit(self.x, sinusoidal_model__fs=self.fs)
    track = model.sinusoidal.tracks_[0]
    self.track_t = np.arange(len(
        track["mag"])) * model.sinusoidal.h / model.sinusoidal.fs
    self.track_a = np.flip(track["mag"]) + 6
    self.track_f = np.flip(track["freq"])

  def test_beat_regression(self):
    """Test BeatRegression"""
    br = beatsdrop.regression.BeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    am_true = self.beat.am(self.track_t)
    am_est, = br.predict(self.track_t, "am")
    self.assert_almost_equal_rmse(am_est, am_true, places=0)

  def test_length_error(self):
    """Test BeatRegression error on short tracks"""
    br = beatsdrop.regression.BeatRegression()
    with self.assertRaises(ValueError):
      br._bounds(self.track_t[:1], None, None, None, None)  # pylint: disable=E1102,W0212

  def test_prediction_length(self):
    """Test BeatRegression prediction array length"""
    br = beatsdrop.regression.BeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    self.assertEqual(br.predict(self.t)[0].size, self.t.size)

  def test_amp_range_no_error(self):
    """Test that constant tracks don't cause errors"""
    br = beatsdrop.regression.BeatRegression()
    b = br._bounds(  # pylint: disable=E1102,W0212
        self.track_t, np.full_like(self.track_a, -np.inf), self.track_f,
        tuple(range(8)), br)
    for (k, v, bk), i in itertools.product(zip(("lower", "upper"), range(2), b),
                                           range(2)):
      with self.subTest(bound=k, partial=i):
        self.assertEqual(bk[i], v)

  def test_dual_beat_regression(self):
    """Test DualBeatRegression"""
    br = beatsdrop.regression.DualBeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    am_true = self.beat.am(self.track_t)
    am_est, = br.predict(self.track_t, "am")
    self.assert_almost_equal_rmse(am_est, am_true, places=1)
    for k, v_est in zip(("a0", "a1", "f0", "f1", "d0", "d1"),
                        beatsdrop.regression.sort_params(br.params_)):
      v_true = getattr(self, k)
      with self.subTest(k=k, true=v_true, est=v_est):
        self.assert_almost_equal_significant(v_true, v_est, places=0)
