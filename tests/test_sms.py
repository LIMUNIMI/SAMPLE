"""Tests related to SMS"""
import unittest
from tests import utils
from sample.sms import sm as sample_sm
from sample.sms import dsp as sample_dsp
import itertools
import numpy as np
import sys
import os
import warnings


class TestSMS(utils.RMSEAssertMixin, unittest.TestCase):
  """Tests related to SMS"""
  sms_pack_repo = "https://gitlab.com/ChromaticIsobar/sms.git"

  @classmethod
  def setUpClass(cls):
    """Download SMS and install it as a package"""
    if not os.path.isfile("sms/__init__.py"):
      os.system(
        "git clone {} sms --recurse-submodules".format(
          cls.sms_pack_repo
        )
      )
      os.system("cd sms && {} setup.py".format(sys.executable))

  @classmethod
  def tearDownClass(cls):
    """Remove SMS"""
    if not os.environ.get("SMS_REPO_PERSISTENT", False):
      os.system("rm -rf sms")

  def setUp(self) -> None:
    """Create a modal-ish sound with three partials for 2 seconds at 44100 Hz"""
    self.fs = 44100
    t_tot = 2
    t = np.linspace(0, t_tot, int(t_tot * self.fs), endpoint=False)
    f = np.array([440, 650, 690])
    a = np.array([1, .5, .45])
    d = np.array([3, 7, 5])

    x = np.squeeze(np.reshape(a, (1, -1)) @ (
      np.exp(-np.reshape(d, (-1, 1)) * np.reshape(t, (1, -1))) *
      np.sin(np.reshape(f, (-1, 1)) * 2 * np.pi * np.reshape(t, (1, -1)))
    ))
    self.x = x / np.max(np.abs(x))
    warnings.filterwarnings(
      "ignore",
      message="numpy.ufunc size changed"
    )

  def tearDown(self):
    """Cleanup after each test"""
    warnings.resetwarnings()

  def test_input_peaks_at_one(self):
    """Test that the input audio peaks at 1"""
    self.assertEqual(np.max(np.abs(self.x)), 1)

  def test_dft(self):
    """Check that DFT computation is consistent"""
    from sms.models import dftModel  # pylint: disable=C0415

    sm = sample_sm.SinusoidalModel()
    sm.w_ = sm.normalized_window
    for i, x_i in enumerate(sm.time_frames(self.x)):
      mx, px = sample_dsp.dft(x_i, sm.w_, sm.n)
      mx_sms, px_sms = dftModel.dftAnal(x_i, sm.w_, sm.n)
      with self.subTest(frame=i, spectrum="magnitude"):
        self.assert_almost_equal_rmse(mx, mx_sms)
      with self.subTest(frame=i, spectrum="phase"):
        self.assert_almost_equal_rmse(px, px_sms)

  def test_peak_detection(self):
    """Check that peak detection is consistent"""
    from sms.models import utilFunctions  # pylint: disable=C0415

    sm = sample_sm.SinusoidalModel()
    sm.w_ = sm.normalized_window
    for i, (mx, _) in enumerate(sm.dft_frames(self.x)):
      ploc = sample_dsp.peak_detect(mx, sm.t)
      ploc_sms = utilFunctions.peakDetection(mx, sm.t)
      for j, (p, p_s) in enumerate(itertools.zip_longest(ploc, ploc_sms)):
        with self.subTest(frame=i, peak_n=j):
          self.assertEqual(p, p_s)


if __name__ == "__main__":
  unittest.main()
