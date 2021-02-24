"""Tests related to SMS"""
import unittest
from tests import utils
from sample.sms import sm as sample_sm
from sample.sms import dsp as sample_dsp
import itertools
import more_itertools
import timeit
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
    """Create a modal-ish sound with three partials for 2 seconds at 44100 Hz.
    Also, initialize sinusoidal model"""
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

    self.sm = sample_sm.SinusoidalModel()
    self.sm.w_ = self.sm.normalized_window

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

    for i, x_i in enumerate(self.sm.time_frames(self.x)):
      mx, px = sample_dsp.dft(x_i, self.sm.w_, self.sm.n)
      mx_sms, px_sms = dftModel.dftAnal(x_i, self.sm.w_, self.sm.n)
      with self.subTest(frame=i, spectrum="magnitude"):
        self.assert_almost_equal_rmse(mx, mx_sms)
      with self.subTest(frame=i, spectrum="phase"):
        self.assert_almost_equal_rmse(px, px_sms)

  def test_peak_detection(self):
    """Check that peak detection is consistent"""
    from sms.models import utilFunctions  # pylint: disable=C0415

    for i, (mx, _) in enumerate(self.sm.dft_frames(self.x)):
      ploc = sample_dsp.peak_detect(mx, self.sm.t)
      ploc_sms = utilFunctions.peakDetection(mx, self.sm.t)
      for j, (p, p_s) in enumerate(itertools.zip_longest(ploc, ploc_sms)):
        with self.subTest(frame=i, peak_n=j):
          self.assertEqual(p, p_s)

  def test_peak_refinement(self):
    """Check that peak interpolation is consistent"""
    from sms.models import utilFunctions  # pylint: disable=C0415

    for i, (mx, px) in enumerate(self.sm.dft_frames(self.x)):
      ploc = sample_dsp.peak_detect(mx, self.sm.t)
      ploc_i, pmag_i, pph_i = sample_dsp.peak_refine(mx, px, ploc)
      ploc_i_sms, pmag_i_sms, pph_i_sms = utilFunctions.peakInterp(mx, px, ploc)
      with self.subTest(frame=i, value="location"):
        self.assert_almost_equal_rmse(ploc_i, ploc_i_sms)
      with self.subTest(frame=i, value="magnitude"):
        self.assert_almost_equal_rmse(pmag_i, pmag_i_sms)
      with self.subTest(frame=i, value="phase"):
        self.assert_almost_equal_rmse(pph_i, pph_i_sms)

  @utils.more_tests
  def test_peak_refinement_speed(self):
    """Check that peak interpolation is faster"""
    from sms.models import utilFunctions  # pylint: disable=C0415

    mx, px = more_itertools.first(self.sm.dft_frames(self.x))
    g = {
      "mx": mx,
      "px": px,
      "ploc": sample_dsp.peak_detect(mx, self.sm.t),
      "sample_dsp": sample_dsp,
      "utilFunctions": utilFunctions,
    }

    def get_time(
      func: str,
    ):
      return timeit.timeit(
        "{}(mx, px, ploc)".format(func),
        number=256,
        globals=g,
      )

    t = get_time("sample_dsp.peak_refine")
    t_sms = get_time("utilFunctions.peakInterp")
    self.assertLessEqual(t, t_sms)
    print("\n  {} <= {}".format(t, t_sms))


if __name__ == "__main__":
  unittest.main()
