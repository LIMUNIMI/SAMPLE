"""Tests for the overall method"""
import unittest
from chromatictools import unittestmixins
import sample
from sample import utils, plots
import numpy as np
import copy
import json


class TestSAMPLE(
  unittestmixins.AssertDoesntRaiseMixin,
  unittest.TestCase
):
  """Tests for the overall method"""
  def setUp(self) -> None:
    """Initialize test audio and SAMPLE model"""
    self.fs = 44100
    self.x = utils.test_audio(fs=self.fs)
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
      s.fit(self.x).predict(np.arange(self.x.size)/self.fs)

  def test_no_exceptions_reverse(self):
    """Test that no exceptions arise from method using reverse mode"""
    s = copy.deepcopy(self.sample).set_params(sinusoidal_model__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x).predict(np.arange(self.x.size)/self.fs)

  def test_sdt_json_serializable(self):
    """Test that SDT parameters are JSON-serializable"""
    with self.assert_doesnt_raise():
      json.dumps(self.sample.fit(self.x).sdt_params_())

  def test_freq_too_high(self):
    """Test track rejection for high frequencies"""
    self.assertFalse(self.st.track_ok(dict(
      freq=np.full(1024, 1e9),
    )))

  def test_freq_too_low(self):
    """Test track rejection for low frequencies"""
    self.assertFalse(self.st.track_ok(dict(
      freq=np.zeros(1024),
    )))

  def test_tracker_reset(self):
    """Test tracker reset"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.subTest(step="chek state is non-empty", var="tracks"):
      self.assertNotEqual(
        len(s.sinusoidal_model.sine_tracker_.tracks_), 0
      )  # pylint: disable=W0212
    with self.subTest(step="chek state is non-empty", var="_active_tracks"):
      self.assertNotEqual(
        len(s.sinusoidal_model.sine_tracker_._active_tracks), 0
      )  # pylint: disable=W0212
    with self.subTest(step="chek state is non-empty", var="_frame"):
      self.assertNotEqual(
        s.sinusoidal_model.sine_tracker_._frame, 0
      )  # pylint: disable=W0212
    s.sinusoidal_model.sine_tracker_.reset()
    with self.subTest(step="chek state is reset", var="tracks"):
      self.assertEqual(
        len(s.sinusoidal_model.sine_tracker_.tracks_), 0
      )  # pylint: disable=W0212
    with self.subTest(step="chek state is reset", var="_active_tracks"):
      self.assertEqual(
        len(s.sinusoidal_model.sine_tracker_._active_tracks), 0
      )  # pylint: disable=W0212
    with self.subTest(step="chek state is reset", var="_frame"):
      self.assertEqual(
        s.sinusoidal_model.sine_tracker_._frame, 0
      )  # pylint: disable=W0212

  def test_refit_deletes_intermediate(self):
    """Test intermediate results reset"""
    s = copy.deepcopy(self.sample).set_params(
      sinusoidal_model__save_intermediate=True
    ).fit(self.x)
    with self.subTest(step="interediate_saved"):
      self.assertTrue(hasattr(
        s.sinusoidal_model, "intermediate_"
      ))
    s.set_params(
      sinusoidal_model__save_intermediate=False
    ).fit(self.x)
    with self.subTest(step="interediate_reset"):
      self.assertFalse(hasattr(
        s.sinusoidal_model, "intermediate_"
      ))

  def test_merge_strategy_raises(self):
    """Test that an unsupported merging strategy causes an Exception"""
    s = copy.deepcopy(self.sample)
    s.set_params(
      sinusoidal_model__merge_strategy="unsupported merging strategy"
    )
    with self.assertRaises(KeyError):
      s.fit(self.x)

  def test_single_merge_strategy(self):
    """Test :data:`"single"` merging strategy"""
    s = copy.deepcopy(self.sample)
    s.set_params(
      sinusoidal_model__merge_strategy="single"
    )
    with self.assert_doesnt_raise():
      s.fit(self.x)

  def test_plot_2d(self):
    """Test 2D plot"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.assert_doesnt_raise():
      plots.sine_tracking_2d(s.sinusoidal_model)

  def test_plot_3d(self):
    """Test 3D plot"""
    s = copy.deepcopy(self.sample).fit(self.x)
    with self.assert_doesnt_raise():
      plots.sine_tracking_3d(s.sinusoidal_model)
