"""Test Beat models"""
import itertools
import unittest

import numpy as np
from chromatictools import unittestmixins
from matplotlib import pyplot as plt

import sample
import sample.beatsdrop.decision
import sample.beatsdrop.regression
import sample.beatsdrop.sample
from sample import plots
import sample.evaluation.metrics

from sklearn import base

bd = sample.beatsdrop


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
      self.assertEqual(bd.regression._get_notnone_attr(c, "c"), "c")  # pylint: disable=W0212
    with self.subTest(test="multiple options"):
      self.assertEqual(
          bd.regression._get_notnone_attr(c, "a", "b", "c"),  # pylint: disable=W0212
          "c")
    with self.subTest(test="multiple invalid options"):
      self.assertEqual(
          bd.regression._get_notnone_attr(  # pylint: disable=W0212
              c, "undefined", "missing", "c"),
          "c")
    with self.subTest(test="only invalid options"):
      with self.assertRaises(AttributeError):
        bd.regression._get_notnone_attr(c, "undefined", "missing")  # pylint: disable=W0212


class TestBeat(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Test beat models"""

  def setUp(self):
    self.fs = 44100
    self.t = np.arange(self.fs) / self.fs

  def test_compute_everything(self):
    """Test that all node variables can be computed"""
    b = bd.Beat()
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)

  def test_compute_everything_modal(self):
    """Test that all node variables can be computed (modal beat)"""
    b = bd.ModalBeat()
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)

  def test_variable_error(self):
    """Test error on undefined variable"""
    b = bd.ModalBeat()
    with self.assertRaises(AttributeError):
      b.yourself(self.t)

  def test_argument_error(self):
    """Test error on time argument undefined"""
    b = bd.ModalBeat()
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
    b = bd.Beat(a0=lambda t: 1 / (1 + t))
    for k in b.variables:
      with self.subTest(variable=k):
        with self.assert_doesnt_raise():
          getattr(b, k)(self.t)


class TestBeatRegression(unittestmixins.AssertDoesntRaiseMixin,
                         unittestmixins.RMSEAssertMixin,
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
    self.d0, self.d1, self.p0, self.p1 = bd.regression.sort_params(
        (self.a0, self.a1, self.f0, self.f1,
         self.d0, self.d1, self.p0, self.p1))
    self.beat = bd.ModalBeat(
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
    self.model = sample.SAMPLE(
        sinusoidal__tracker__max_n_sines=32,
        sinusoidal__tracker__reverse=True,
        sinusoidal__t=-90,
        sinusoidal__intermediate__save=True,
        sinusoidal__tracker__peak_threshold=-45,
        sinusoidal__padded=True,
    ).fit(self.x, sinusoidal__tracker__fs=self.fs)
    track = self.model.sinusoidal.tracks_[0]
    self.track_t = np.arange(len(
        track["mag"])) * self.model.sinusoidal.h / self.model.sinusoidal.fs
    self.track_a = np.flip(track["mag"]) + 6
    self.track_f = np.flip(track["freq"])

  def test_beat_regression(self):
    """Test BeatRegression"""
    br = bd.regression.BeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    am_true = self.beat.am(self.track_t)
    am_est, = br.predict(self.track_t, "am")
    self.assert_almost_equal_rmse(am_est, am_true, places=0)

  def test_length_error(self):
    """Test BeatRegression error on short tracks"""
    br = bd.regression.BeatRegression()
    with self.assertRaises(ValueError):
      br.bounds(self.track_t[:1], None, None, None, None)

  def test_feasibility_error(self):
    """Test BeatRegression error on unfeasible problems"""
    br = bd.regression.BeatRegression(bounds=lambda t, a, f, p, m: (
        (p[0] + 1, p[1] - 2, *p[2:]), (p[0] + 2, p[1] - 1, *p[2:])))
    with self.assertRaises(ValueError):
      br.fit(t=self.track_t, a=self.track_a, f=self.track_f)

  def test_prediction_length(self):
    """Test BeatRegression prediction array length"""
    br = bd.regression.BeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    self.assertEqual(br.predict(self.t)[0].size, self.t.size)

  def test_amp_range_no_error(self):
    """Test that constant tracks don't cause errors"""
    br = bd.regression.BeatRegression()
    b = br.bounds(self.track_t, np.full_like(self.track_a, -np.inf),
                  self.track_f, tuple(range(8)), br)
    for (k, v, bk), i in itertools.product(zip(("lower", "upper"), range(2), b),
                                           range(2)):
      with self.subTest(bound=k, partial=i):
        self.assertEqual(bk[i], v)

  def test_dual_beat_regression(self):
    """Test DualBeatRegression"""
    br = bd.regression.DualBeatRegression()
    br.fit(t=self.track_t, a=self.track_a, f=self.track_f)
    am_true = self.beat.am(self.track_t)
    am_est, = br.predict(self.track_t, "am")
    self.assert_almost_equal_rmse(am_est, am_true, places=1)
    for k, v_est in zip(("a0", "a1", "f0", "f1", "d0", "d1"),
                        bd.regression.sort_params(br.params_)):
      v_true = getattr(self, k)
      with self.subTest(k=k, true=v_true, est=v_est):
        self.assert_almost_equal_significant(v_true, v_est, places=0)

  def test_comparison_plot(self):
    """Test comparison plot"""
    with self.assert_doesnt_raise():
      plots.beatsdrop_comparison(
          self.model, {
              "BeatsDROP": bd.regression.DualBeatRegression(),
              "Baseline": bd.regression.BeatRegression(),
          }, self.x)

  def test_comparison_plot_ht(self):
    """Test comparison plot with Hilbert transform"""
    with self.assert_doesnt_raise():
      plots.beatsdrop_comparison(self.model, {
          "BeatsDROP": bd.regression.DualBeatRegression(),
          "Baseline": bd.regression.BeatRegression(),
      },
                                 self.x,
                                 signal_hilbert_am=True)

  def test_comparison_plot_ht_sub(self):
    """Test comparison plot with Hilbert transform (subsampled)"""
    with self.assert_doesnt_raise():
      plots.beatsdrop_comparison(self.model, {
          "BeatsDROP": bd.regression.DualBeatRegression(),
          "Baseline": bd.regression.BeatRegression(),
      },
                                 self.x,
                                 signal_hilbert_am=256)

  def test_comparison_plot_t(self):
    """Test comparison plot (transposed)"""
    with self.assert_doesnt_raise():
      plots.beatsdrop_comparison(self.model, {
          "BeatsDROP": bd.regression.DualBeatRegression(),
          "Baseline": bd.regression.BeatRegression(),
      },
                                 self.x,
                                 transpose=True)

  def test_comparison_plot_axs(self):
    """Test comparison plot (specify axs)"""
    _, axs = plt.subplots(3, 2)
    with self.assert_doesnt_raise():
      plots.beatsdrop_comparison(self.model, {
          "BeatsDROP": bd.regression.DualBeatRegression(),
          "Baseline": bd.regression.BeatRegression(),
      },
                                 self.x,
                                 axs=axs)

  def test_sample_dummy(self):
    """Test SAMPLE+BeatsDROP integration with a dummy decisor"""
    model_bd = bd.sample.SAMPLEBeatsDROP(beat_decisor=bd.decision.BeatDecisor(),
                                         **base.clone(self.model).get_params())
    model_bd.fit(self.x)
    with self.subTest(test="freqs"):
      np.testing.assert_array_equal(model_bd.freqs_, self.model.freqs_)
    with self.subTest(test="amps"):
      np.testing.assert_array_equal(model_bd.amps_, self.model.amps_)
    with self.subTest(test="decays"):
      np.testing.assert_array_equal(model_bd.decays_, self.model.decays_)

  def test_sample(self):
    """Test SAMPLE+BeatsDROP integration"""
    model_bd = bd.sample.SAMPLEBeatsDROP(beat_decisor__intermediate__save=True,
                                         **base.clone(self.model).get_params())
    model_bd.fit(self.x)
    # Check tracks are identical
    with self.subTest(check="n_tracks"):
      self.assertEqual(len(model_bd.sinusoidal.tracks_),
                       len(self.model.sinusoidal.tracks_))
    for i, (t, u) in enumerate(
        zip(model_bd.sinusoidal.tracks_, self.model.sinusoidal.tracks_)):
      with self.subTest(check_track=i):
        self.assertEqual(t.keys(), u.keys())
        for k in t:
          with self.subTest(check_track=i, k=k):
            np.testing.assert_array_equal(t[k], u[k])
    # Check parameters are different
    with self.subTest(has_phases=True):
      self.assertEqual(model_bd.param_matrix_.shape[0], 4)
    with self.subTest(found_beats=True):
      self.assertGreater(model_bd.param_matrix_.shape[1],
                         self.model.param_matrix_.shape[1])
    n_beats = sum(model_bd.beat_decisor.intermediate["decision"])
    with self.subTest(found_beats=n_beats):
      self.assertEqual(model_bd.param_matrix_.shape[1] - n_beats,
                       self.model.param_matrix_.shape[1])
    # Check resynthesis is better
    y = self.model.predict(self.t)
    y_bd = model_bd.predict(self.t)
    closs = sample.evaluation.metrics.CochleagramLoss(fs=self.fs, stride=1 << 9)
    with self.subTest(check="lower_loss"):
      self.assertLess(closs(self.x, y_bd), closs(self.x, y))

  def test_sample_parallel_fit(self):
    """Test SAMPLE+BeatsDROP integration in multiprocessing"""
    model_bd = bd.sample.SAMPLEBeatsDROP(beat_decisor__intermediate__save=True,
                                         **base.clone(self.model).get_params())
    model_bd.fit(self.x)
    model_bdp = base.clone(model_bd).fit(self.x, n_jobs=4)
    with self.subTest(test="freqs"):
      np.testing.assert_array_equal(model_bd.freqs_, model_bdp.freqs_)
    with self.subTest(test="amps"):
      np.testing.assert_array_equal(model_bd.amps_, model_bdp.amps_)
    with self.subTest(test="decays"):
      np.testing.assert_array_equal(model_bd.decays_, model_bdp.decays_)
    with self.subTest(test="phases"):
      np.testing.assert_array_equal(model_bd.phases_, model_bdp.phases_)
    with self.subTest(test="correlation tests"):
      self.assertTrue("test" in model_bd.beat_decisor.intermediate.get_state())
      self.assertTrue("test" in model_bdp.beat_decisor.intermediate.get_state())
      np.testing.assert_array_equal(model_bd.beat_decisor.intermediate["test"],
                                    model_bdp.beat_decisor.intermediate["test"])
