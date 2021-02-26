"""Tests for the overall method"""
import unittest
import sample
from sample import utils
import json


class TestSAMPLE(unittest.TestCase):
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
    exceptions = False
    msg = ""
    try:
      self.sample.fit(self.x)
    except Exception as e:  # pylint: disable=W0703
      exceptions = True
      msg = str(e)
    self.assertFalse(exceptions, msg=msg)

  def test_sdt_json_serializable(self):
    """Test that SDT parameters are JSON-serializable"""
    exceptions = False
    msg = ""
    try:
      self.sample.fit(self.x)
      json.dumps(self.sample.sdt_params_, indent=2)
    except Exception as e:  # pylint: disable=W0703
      exceptions = True
      msg = str(e)
    self.assertFalse(exceptions, msg=msg)
