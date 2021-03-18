"""Tests for the overall method"""
import unittest
from chromatictools import unittestmixins
import sample
from sample import utils
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

  def test_no_exceptions(self):
    """Test that no exceptions arise from method"""
    with self.assert_doesnt_raise():
      self.sample.fit(self.x).predict(np.arange(self.x.size)/self.fs)

  def test_no_exceptions_reverse(self):
    """Test that no exceptions arise from method using reverse mode"""
    s = copy.deepcopy(self.sample).set_params(sinusoidal_model__reverse=True)
    with self.assert_doesnt_raise():
      s.fit(self.x).predict(np.arange(self.x.size)/self.fs)

  def test_sdt_json_serializable(self):
    """Test that SDT parameters are JSON-serializable"""
    with self.assert_doesnt_raise():
      json.dumps(self.sample.fit(self.x).sdt_params_())
