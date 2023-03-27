"""Tests for psychoacoustic models"""
import itertools
import unittest

import numpy as np
import sklearn.exceptions
from chromatictools import unittestmixins
from sample import plots, psycho
from sample.evaluation import random
from sample.utils import dsp as dsp_utils

import tests.utils as test_utils


class TestPsycho(unittestmixins.AssertDoesntRaiseMixin,
                 unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Tests psychoacoustic models functions"""

  @test_utils.coherence_check_method(
      fwd=psycho.hz2bark,
      bak=psycho.bark2hz,
      modes=("unsupported", "zwicker", "traunmuller", "wang"),
      no_fwd=("unsupported",),
      no_bak=("zwicker",),
  )
  def test_bark(self):
    """Test coherence of conversion for Barks"""
    pass

  @test_utils.coherence_check_method(fwd=psycho.hz2mel,
                                     bak=psycho.mel2hz,
                                     modes=("unsupported", "default", "fant"),
                                     no_fwd=("unsupported",))
  def test_mel(self):
    """Test coherence of conversion for Mels"""
    pass

  @test_utils.coherence_check_method(fwd=psycho.hz2cams,
                                     bak=psycho.cams2hz,
                                     mode_key="degree",
                                     modes=("unsupported", "linear",
                                            "quadratic"),
                                     no_fwd=("unsupported",))
  def test_cams(self):
    """Test coherence of conversion for ERB-rate scale"""
    pass

  def test_erb_monotonicity(self):
    """Test monotonicity of ERBs"""
    f = np.linspace(0, 2e4, 1024)
    for deg, use_buf in itertools.product(("linear", "quadratic"),
                                          (False, True)):
      kw = {} if use_buf else {"out": np.empty_like(f)}
      kw["degree"] = deg
      erbs = psycho.erb(f, **kw)
      with self.subTest(degree=deg, buffer=use_buf):
        self.assertTrue(np.greater(np.diff(erbs), 0).all())


class TestTF(unittestmixins.AssertDoesntRaiseMixin,
             unittestmixins.RMSEAssertMixin, unittest.TestCase):
  """Test time-frequency representations"""

  def setUp(self):
    """Setup test audio"""
    self.x, self.fs, _ = random.BeatsGenerator(seed=1234).audio()
    self.stft_kws = {
        "nperseg": 2048,
        "noverlap": 1024,
        "window": "hamming",
        "fs": self.fs
    }

  def test_cochleagram_shape(self):
    """Test cochleagram shape"""
    for method in ("auto", "fft", "direct", "overlap-add"):
      coch, freqs = psycho.cochleagram(self.x,
                                       fs=self.fs,
                                       normalize=True,
                                       method=method)
      with self.subTest(shape="rows", method=method):
        self.assertEqual(coch.shape[0], np.size(freqs))
      with self.subTest(shape="cols", method=method):
        self.assertGreater(coch.shape[1], self.x.size)

  def test_cochleagram_twostep_shape(self):
    """Test cochleagram shape from :func:`GammatoneFilterbank.convolve`"""
    gtfb = psycho.GammatoneFilterbank(normalize=True)
    coch = gtfb.convolve(self.x, fs=self.fs)
    with self.subTest(shape="rows"):
      self.assertEqual(coch.shape[0], len(gtfb))
    with self.subTest(shape="cols"):
      self.assertGreater(coch.shape[1], self.x.size)

  def test_error_complex_irs(self):
    """Test raising exception when IRs are analytic"""
    for a in (False, True):
      irs = psycho.GammatoneFilterbank().precompute(fs=self.fs, analytic=a)
      for arg in (None, "ir", "input", "output"):
        with self.subTest(ir_analytic=a, arg=arg):
          if a and arg in ("input", "output"):
            with self.assertRaises(ValueError):
              psycho.cochleagram(self.x, filterbank=irs, analytic=arg)
          else:
            with self.assert_doesnt_raise():
              psycho.cochleagram(self.x, filterbank=irs, analytic=arg)

    gtfb = psycho.GammatoneFilterbank(normalize=True)
    coch = gtfb.convolve(self.x, fs=self.fs)
    with self.subTest(shape="rows"):
      self.assertEqual(coch.shape[0], len(gtfb))
    with self.subTest(shape="cols"):
      self.assertGreater(coch.shape[1], self.x.size)

  def test_cochleagram_sorted(self):
    """Test cochleagram frequencies are sorted"""
    gtfb = psycho.GammatoneFilterbank()
    for i in range(len(gtfb) - 1):
      with self.subTest(f"{i}/{len(gtfb)} < {i+1}/{len(gtfb)}"):
        self.assertLess(gtfb[i].f, gtfb.f[i + 1])

  def test_cochleagram_bandwidth_fn(self):
    """Test cochleagram bandwidth function"""
    gtfb = psycho.GammatoneFilterbank(bandwidth=lambda _: 1)
    for gtf in gtfb:
      with self.subTest(f_c=gtf.f):
        self.assertEqual(gtf.bandwidth, 1)

  def test_cochleagram_bandwidth_c(self):
    """Test cochleagram bandwidth constant"""
    gtfb = psycho.GammatoneFilterbank(bandwidth=1)
    for gtf in gtfb:
      with self.subTest(f_c=gtf.f, bandwidth="init"):
        self.assertEqual(gtf.bandwidth, 1)
    for gtf in gtfb:
      gtf.bandwidth = 10
      with self.subTest(f_c=gtf.f, bandwidth="set"):
        self.assertEqual(gtf.bandwidth, 10)

  def test_cochleagram_bandwidth_t_c_vs_delay(self):
    """Test cochleagram setting leading time vs group delay"""
    gtfb = psycho.GammatoneFilterbank()
    for gtf in gtfb:
      gtf.t_c = 0
      with self.subTest(f_c=gtf.f, t_c=0):
        self.assertGreater(gtf.group_delay, 0)
        self.assertEqual(gtf.t_c, 0)
      d = gtf.group_delay
      gtf.t_c = lambda _: 0
      with self.subTest(f_c=gtf.f, t_c="function() = 0"):
        self.assertEqual(gtf.group_delay, d)
        self.assertEqual(gtf.t_c, 0)
      d = gtf.group_delay
      gtf.group_delay = 0
      with self.subTest(f_c=gtf.f, group_delay=0):
        self.assertEqual(gtf.group_delay, 0)
        self.assertEqual(gtf.t_c, d)

  def test_cochleagram_phi_c(self):
    """Test cochleagram gammatone phase"""
    gtfb = psycho.GammatoneFilterbank(phi=0)
    for gtf in gtfb:
      with self.subTest(phase="constant"):
        self.assertEqual(gtf.phi, 0)
      gtf.phi = lambda _: 0
      with self.subTest(phase="function"):
        self.assertEqual(gtf.phi, 0)

  def test_cochleagram_a_weighting(self):
    """Test cochleagram A-weighting"""
    gtfb = psycho.GammatoneFilterbank(
        a=lambda gtf: dsp_utils.db2a(psycho.a_weighting(gtf.f)))
    for gtf in gtfb:
      with self.subTest(f_c=gtf.f):
        self.assertAlmostEqual(dsp_utils.a2db(gtf.a), psycho.a_weighting(gtf.f))

  def test_cochleagram_a_c(self):
    """Test cochleagram constant amplitude"""
    gtfb = psycho.GammatoneFilterbank()
    for gtf in gtfb:
      gtf.a = 1
      with self.subTest(f_c=gtf.f):
        self.assertEqual(gtf.a, 1)

  def test_gammatone_t60_error_false_start(self):
    """Test that gammatone filter raises error on bad start"""
    gtf = psycho.GammatoneFilter(n=100)
    with self.assertRaises(ValueError):
      gtf.t60(initial_range=1e-6)

  def test_gammatone_t60_error_convergence(self):
    """Test that gammatone filter raises error on bad convergence"""
    gtf = psycho.GammatoneFilter(n=100)
    with self.assertRaises(sklearn.exceptions.ConvergenceWarning):
      gtf.t60(initial_range=1e-3, steps=4)

  def test_cochleagram_rms_peak(self):
    """Test cochleagram RMS peak"""
    irs = psycho.GammatoneFilterbank(normalize=True).precompute(fs=self.fs)
    t = np.arange(int(self.fs * 0.125)) / self.fs
    x = np.empty_like(t)
    for stride in (None, int(self.fs * 0.001)):
      for i, f in enumerate(irs.freqs):
        np.multiply(2 * np.pi * f, t, out=x)
        np.sin(x, out=x)
        coch, _ = psycho.cochleagram(x, filterbank=irs, stride=stride)
        coch_rms = np.sqrt(np.mean(np.square(coch), axis=1))
        with self.subTest(stride=stride, i=i, f=f):
          self.assertEqual(coch_rms.size, len(irs))
          self.assertEqual(i, np.argmax(coch_rms))

  def test_cochleagram_direct_vs_strided(self):
    """Test stride cochleagram correctness (against direct)"""
    for analytic in (True, False):
      coch, _ = psycho.cochleagram(self.x,
                                   fs=self.fs,
                                   normalize=True,
                                   analytic=analytic,
                                   method="direct")
      for stride in (0, 0.005, 0.010):
        stride = max(1, int(stride * self.fs))
        coch_s, _ = psycho.cochleagram(self.x,
                                       fs=self.fs,
                                       normalize=True,
                                       analytic=analytic,
                                       stride=stride)
        coch_d = coch[:, ::stride]
        with self.subTest(analytic=analytic, stride=stride, what="shape"):
          self.assertEqual(coch_d.shape, coch_s.shape)
        with self.subTest(analytic=analytic, stride=stride, what="rmse"):
          self.assert_almost_equal_rmse(coch_s, coch_d, places=5)

  def test_cochleagram_rms_peak_analytic(self):
    """Test complex cochleagram RMS peak"""
    irs = psycho.GammatoneFilterbank(normalize=True).precompute(fs=self.fs,
                                                                analytic=True)
    t = np.arange(int(self.fs * 0.125)) / self.fs
    x = np.empty_like(t)
    for stride in (None, int(self.fs * 0.001)):
      for i, f in enumerate(irs.freqs):
        np.multiply(2 * np.pi * f, t, out=x)
        np.sin(x, out=x)
        coch, _ = psycho.cochleagram(x, filterbank=irs, stride=stride)
        coch_rms = np.sqrt(np.mean(np.square(np.abs(coch)), axis=1))
        with self.subTest(stride=stride, i=i, f=f):
          self.assertEqual(coch_rms.size, len(irs))
          self.assertEqual(i, np.argmax(coch_rms))

  def test_ir_error_no_fs(self):
    """Test that exception is raised if no fs is specified"""
    with self.assertRaises(ValueError):
      psycho.GammatoneFilter(f=200).ir()

  def test_cochleagram_error_no_fs(self):
    """Test that exception is raised if no fs is specified"""
    with self.assertRaises(TypeError):
      psycho.cochleagram(self.x)

  def test_ir_no_error_time(self):
    """Test that no exception is raised if only one time-step"""
    with self.assert_doesnt_raise():
      psycho.GammatoneFilter(f=200, normalize=True).ir(t=[0.0])

  def test_gtfb_error_getattr(self):
    """Test that exception is raised if accessing wrong attributes"""
    with self.assertRaises(AttributeError):
      psycho.GammatoneFilterbank().invalid_attribute  # pylint: disable=W0106

  def test_ir_size(self):
    """Test that ir size is coherent"""
    gtfb = psycho.GammatoneFilterbank()
    for f in gtfb:
      with self.subTest(f_c=f.f):
        self.assertEqual(f.ir_size(fs=self.fs), f.ir(fs=self.fs).size)

  def test_ir_normalized(self):
    """Test that IR normalization is coherent"""
    gtfb = psycho.GammatoneFilterbank(normalize=True)
    for f in gtfb:
      with self.subTest(f_c=f.f):
        self.assertAlmostEqual(np.sum(np.square(f.ir(fs=self.fs))), 1)

  def test_cochleagram_plot(self):
    """Test plotting cochleagram"""
    coch, freqs = psycho.cochleagram(
        self.x,
        fs=self.fs,
        postprocessing=lambda c: dsp_utils.complex2db(dsp_utils.normalize(c),
                                                      floor=1e-3))

    plots.tf_plot(coch,
                  flim=psycho.hz2cams(freqs[[0, -1]]),
                  tlim=(0, self.x.size / self.fs),
                  cmap="afmhot")
    plots.plt.clf()

  def test_cochleagram_error_method_stride(self):
    """Test that cochleagram raises an error if
    both method and stride are specified"""
    with self.assertRaises(ValueError):
      psycho.cochleagram(self.x,
                         fs=self.fs,
                         method="overlap-add",
                         stride=int(self.fs * 0.005))

  def test_cochleagram_error_complex_strided(self):
    """Test that cochleagram raises an error for
    strided convolution of complex input"""
    with self.assertRaises(ValueError):
      psycho.cochleagram(self.x,
                         fs=self.fs,
                         analytic="input",
                         stride=int(self.fs * 0.005))

  def test_mel_spectrogram_shape(self, n_filters: int = 81):
    """Test mel-spectrogram shape"""
    freqs, _, melspec = psycho.mel_spectrogram(self.x,
                                               n_filters=n_filters,
                                               stft_kws=self.stft_kws)
    with self.subTest(shape="rows"):
      self.assertEqual(melspec.shape[0], np.size(freqs))
      self.assertEqual(melspec.shape[0], n_filters)

  def test_mel_spectrogram_shape_bw(self, n_filters: int = 81):
    """Test mel-spectrogram shape when using custom bandwidth"""
    freqs, _, melspec = psycho.mel_spectrogram(self.x,
                                               n_filters=n_filters,
                                               stft_kws=self.stft_kws,
                                               bandwidth=psycho.erb)
    with self.subTest(shape="rows"):
      self.assertEqual(melspec.shape[0], np.size(freqs))
      self.assertEqual(melspec.shape[0], n_filters)

  def test_mel_spectrogram_shape_twostep(self, n_filters: int = 81):
    """Test mel-spectrogram shape by providing filterbank"""
    n = self.stft_kws["nperseg"]
    freqs = np.arange(n // 2 + 1 - n % 2) * self.fs / n
    fbank, mfreqs = psycho.mel_triangular_filterbank(freqs=freqs,
                                                     n_filters=n_filters)
    freqs_none, _, melspec = psycho.mel_spectrogram(self.x,
                                                    stft_kws=self.stft_kws,
                                                    filterbank=fbank)
    with self.subTest(freq=None):
      self.assertIsNone(freqs_none)
    with self.subTest(shape="rows"):
      self.assertEqual(melspec.shape[0], mfreqs.size)
      self.assertEqual(melspec.shape[0], n_filters)

  def test_mel_spectrogram_flim(self):
    """Test mel-spectrogram manually provided frequency limits"""
    flim = np.power(2, np.linspace(np.log2(20), np.log2(20000), 34))
    freqs, _, melspec = psycho.mel_spectrogram(self.x,
                                               flim=flim,
                                               stft_kws=self.stft_kws)
    with self.subTest(freqs="shape"):
      self.assertEqual(melspec.shape[0], np.size(freqs))
      self.assertEqual(melspec.shape[0], flim.size - 2)

    for i, (f, g) in enumerate(zip(flim[1:-1], freqs)):
      with self.subTest(freq=i):
        self.assertEqual(f, g)

  def test_mel_spectrogram_error_notenough(self):
    """Test cochleagram error when filters are <= 0"""
    with self.assertRaises(ValueError):
      psycho.mel_spectrogram(self.x, stft_kws=self.stft_kws)

  def test_mel_spectrogram_plot(self, n_filters: int = 81):
    """Test plotting mel-spectrogram"""
    freqs, times, melspec = psycho.mel_spectrogram(self.x,
                                                   n_filters=n_filters,
                                                   stft_kws=self.stft_kws)
    plots.tf_plot(dsp_utils.complex2db(dsp_utils.normalize(melspec),
                                       floor=1e-3),
                  flim=psycho.hz2mel(freqs[[0, -1]]),
                  tlim=times[[0, -1]],
                  cmap="afmhot")
    plots.plt.clf()
