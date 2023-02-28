"""Tests related to SMS"""
import itertools
import os
import sys
import timeit
import unittest
import warnings

import more_itertools
from chromatictools import unittestmixins
from sample.evaluation import random
from sample.sms import dsp as sample_dsp
from sample.sms import sm as sample_sm

from tests import utils


class TestSMS(unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests related to SMS"""
  sms_pack_repo = "https://gitlab.com/ChromaticIsobar/sms.git"

  @classmethod
  def setUpClass(cls):
    """Download SMS and install it as a package"""
    if not os.path.isfile("sms/__init__.py"):
      os.system(f"git clone {cls.sms_pack_repo} sms --recurse-submodules")
      os.system(f"cd sms && {sys.executable} setup.py")

  @classmethod
  def tearDownClass(cls):
    """Remove SMS"""
    if not os.environ.get("SMS_REPO_PERSISTENT", False):
      os.system("rm -rf sms")

  def setUp(self) -> None:
    """Initialize test audio and sinusoidal model"""
    self.x, self.fs, _ = random.BeatsGenerator(seed=1234).audio()
    self.sm = sample_sm.SinusoidalModel()
    self.sm.w_ = self.sm._normalized_window  # pylint: disable=W0212

    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

  def tearDown(self):
    """Cleanup after each test"""
    warnings.resetwarnings()

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
      ploc_i, pmag_i, pph_i = sample_dsp.peak_refine(ploc, mx, px)  # pylint: disable=W0632
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

    def get_time(func: str,):
      return timeit.timeit(
          func,
          number=256,
          globals=g,
      )

    t = get_time("sample_dsp.peak_refine(ploc, mx, px)")
    t_sms = get_time("utilFunctions.peakInterp(mx, px, ploc)")
    self.assertLessEqual(t, t_sms)
    print("\n" + f"  {t} <= {t_sms}")

  def test_intermediate(self):
    """Check that intermediate results are saved"""
    self.sm.fit(self.x, intermediate__save=True)
    self.assertTrue(hasattr(self.sm.intermediate, "cache_"))

    for k in (
        "stft",
        "peaks",
    ):
      with self.subTest(key=k):
        self.assertTrue(k in self.sm.intermediate.cache_, "Key not found")
        self.assertGreater(len(self.sm.intermediate[k]), 0, "List is empty")


if __name__ == "__main__":
  unittest.main()
