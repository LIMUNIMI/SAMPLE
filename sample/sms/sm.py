"""Sinusoidal model"""
from sample.sms import dsp
import numpy as np
from sklearn import base
import functools
from typing import Optional, Tuple, Generator, Iterable


class SinusoidalModel(base.TransformerMixin, base.BaseEstimator):
  """Model for sinusoidal tracking

  Args:
    fs (int): sampling frequency in Hz. Defaults to 44100
    w: Analysis window. Defaults to None (if None,
      the :func:`default_window` is used)
    n (int): FFT size. Defaults to 2048
    h (int): Window hop size. Defaults to 500
    t (float): threshold in dB. Defaults to -90
    max_n_sines (int): Maximum number of tracks per frame. Defaults to 100
    min_sine_dur (float): Minimum duration of a track in seconds.
      Defaults to 0.01
    freq_dev_offset (float): Frequency deviation threshold at 0Hz.
      Defaults to 20
    freq_dev_slope (float): Slope of frequency deviation threshold.
      Defaults to 0.01
    save_intermediate (bool): If True, save intermediate data structures in
      the attribute :py:data:`intermediate_`. Defaults to False

  Attributes:
    w_ (array): Effective analysis window
    intermediate_ (dict): Dictionary of intermediate data structures"""
  def __init__(
    self,
    fs: int = 44100,
    w: Optional[np.ndarray] = None,
    n: int = 2048,
    h: int = 500,
    t: float = -90,
    max_n_sines: int = 100,
    min_sine_dur: float = 0.01,
    freq_dev_offset: float = 20,
    freq_dev_slope: float = 0.01,
    save_intermediate: bool = False,
  ):
    self.fs = fs
    self.w = w
    self.n = n
    self.h = h
    self.t = t
    self.max_n_sines = max_n_sines
    self.min_sine_dur = min_sine_dur
    self.freq_dev_offset = freq_dev_offset
    self.freq_dev_slope = freq_dev_slope
    self.save_intermediate = save_intermediate

  def fit(
    self,
    x: np.ndarray,
    y=None,  # pylint: disable=W0613
    **kwargs
  ):
    """Analyze audio data

    Args:
      x (array): audio input
      y (ignored): exists for compatibility
      kwargs: Any parameter, overrides initialization
    """
    self.set_params(**kwargs)
    self.w_ = self.normalized_window

    for mx, px in map(
      functools.partial(self.intermediate, "stft"),
      self.dft_frames(x)
    ):
      _, _, _ = self.intermediate(
        "peaks",
        dsp.peak_detect_interp(mx, px, self.t)
      )

  def intermediate(self, key: str, value):
    """Save intermediate results if :py:data:`save_intermediate` is True

    Arguments:
      key (str): Data name
      value: Data

    Returns:
      object: The input value"""
    if self.save_intermediate:
      if not hasattr(self, "intermediate_"):
        self.intermediate_ = {}
      if key not in self.intermediate_:
        self.intermediate_[key] = []
      self.intermediate_[key].append(value)
    return value

  @property
  def default_window(self) -> np.ndarray:
    """Default window (2001 samples Hamming window)"""
    return np.hamming(2001)

  @property
  def normalized_window(self) -> np.ndarray:
    """Normalized analysis window (if None, normalized default window)"""
    w = self.default_window if self.w is None else self.w
    return w / np.sum(w)

  def pad_input(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad input at the beginning (so that the first window is centered at
    the first sample) and at the end (to analyze all samples)

    Args:
      x (array): The input array

    Returns:
      (array, int): The padded array and the initial padding length"""
    a = (self.w_.size + 1) // 2
    b = self.w_.size // 2
    y = np.zeros(x.size + a + b)
    y[a:(a + x.size)] = x
    return y, a

  def time_frames(self, x: np.ndarray) -> Generator[np.ndarray, None, None]:
    """Generator of frames for a given input

    Args:
      x (array): Input

    Returns:
      generator: Generator of overlapping frames of the padded input"""
    y, a = self.pad_input(x)
    for i in range(a, y.size - a, self.h):
      yield y[(i - a):(i - a + self.w_.size)]

  def dft_frames(
    self,
    x: np.ndarray
  ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Iterable of DFT frames for a given input

    Args:
      x (array): Input

    Returns:
      iterable: Iterable of overlapping DFT frames (magnitude and phase)
      of the padded input"""
    return map(
      functools.partial(dsp.dft, w=self.w_, n=self.n),
      self.time_frames(x)
    )
