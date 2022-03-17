"""Tests for psychoacoustic models"""
import itertools
import unittest

import numpy as np
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
      kw = {} if use_buf else dict(out=np.empty_like(f))
      kw["degree"] = deg
      erbs = psycho.erb(f, **kw)
      with self.subTest(degree=deg, buffer=use_buf):
        self.assertTrue(np.greater(np.diff(erbs), 0).all())


class TestTF(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Test time-frequency representations"""

  def setUp(self):
    """Setup test audio"""
    self.x, self.fs, _ = random.BeatsGenerator(seed=1234).audio()
    self.stft_kws = dict(nperseg=2048,
                         noverlap=1024,
                         window="hamming",
                         fs=self.fs)

  def test_cochleagram_shape(self, n_filters: int = 81, size: float = 1 / 16):
    """Test cochleagram shape"""
    size = int(size * self.fs)
    coch, freqs = psycho.cochleagram(self.x,
                                     n_filters=n_filters,
                                     size=size,
                                     fs=self.fs)
    with self.subTest(shape="rows"):
      self.assertEqual(coch.shape[0], np.size(freqs))
      self.assertEqual(coch.shape[0], n_filters)
    with self.subTest(shape="cols"):
      # default is full convolution
      self.assertEqual(coch.shape[1], self.x.size + size - 1)

  def test_cochleagram_shape_twostep(self,
                                     n_filters: int = 81,
                                     size: float = 1 / 16):
    """Test cochleagram shape by providing filterbank"""
    size = int(size * self.fs)
    filterbank, freqs = psycho.gammatone_filterbank(n_filters=n_filters,
                                                    size=size,
                                                    fs=self.fs)
    coch, freqs_none = psycho.cochleagram(self.x,
                                          filterbank=filterbank,
                                          convolve_kws=dict(mode="same"))
    with self.subTest(freq=None):
      self.assertIsNone(freqs_none)
    with self.subTest(shape="rows"):
      self.assertEqual(coch.shape[0], freqs.size)
      self.assertEqual(coch.shape[0], n_filters)
    with self.subTest(shape="cols"):
      # same-size convolution
      self.assertEqual(coch.shape[1], self.x.size)

  def test_cochleagram_error_undef(self):
    """Test cochleagram error when filter size is undefined"""
    with self.assertRaises(ValueError):
      psycho.cochleagram(self.x, fs=self.fs)

  def test_cochleagram_retry_nonlin(self,
                                    n_filters: int = 81,
                                    size: float = 1 / 16):
    """Test that nonlinearity is first tried with a "out" parameter"""
    size = int(size * self.fs)

    class NonLinearity:
      """Non-linear dummy function for test"""

      def __init__(self):
        self.called_out = False
        self.called_no_out = False

      def __call__(self, a, *args, **kwargs):
        if "out" in kwargs:
          self.called_out = True
          raise TypeError
        self.called_no_out = True
        return np.square(a, *args, **kwargs)

    nonlinearity = NonLinearity()
    with self.assert_doesnt_raise():
      psycho.cochleagram(self.x,
                         fs=self.fs,
                         nonlinearity=nonlinearity,
                         n_filters=n_filters,
                         size=size)
    self.assertTrue(nonlinearity.called_out)
    self.assertTrue(nonlinearity.called_no_out)

  def test_cochleagram_rms_peak(self,
                                n_filters: int = 40,
                                size: float = 1 / 16):
    """Test cochleagram RMS peak"""
    size = int(size * self.fs)
    filterbank, freqs = psycho.gammatone_filterbank(n_filters=n_filters,
                                                    freqs=(20, 20000),
                                                    size=size,
                                                    fs=self.fs,
                                                    a_norm=True)
    t = np.arange(int(self.fs * 0.125)) / self.fs
    x = np.empty_like(t)
    for i, f in enumerate(freqs):
      np.multiply(2 * np.pi * f, t, out=x)
      np.sin(x, out=x)
      coch, _ = psycho.cochleagram(x,
                                   filterbank=filterbank,
                                   convolve_kws=dict(mode="same"))
      coch_rms = np.sqrt(np.mean(np.square(coch), axis=1))
      with self.subTest(i=i, f=f):
        self.assertEqual(coch_rms.size, np.size(freqs))
        self.assertEqual(i, np.argmax(coch_rms))

  def test_cochleagram_plot(self, n_filters: int = 81, size: float = 1 / 16):
    """Test plotting cochleagram"""
    size = int(size * self.fs)
    coch, freqs = psycho.cochleagram(self.x,
                                     n_filters=n_filters,
                                     size=size,
                                     fs=self.fs,
                                     convolve_kws=dict(mode="same"))
    plots.tf_plot(dsp_utils.complex2db(dsp_utils.normalize(coch), floor=1e-3),
                  flim=psycho.hz2cams(freqs[[0, -1]]),
                  tlim=(0, self.x.size / self.fs),
                  cmap="afmhot")
    plots.plt.clf()

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
