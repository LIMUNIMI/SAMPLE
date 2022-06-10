"""Random generation of modal-like sounds"""
import functools
from typing import Optional

import numpy as np
import sample.sample
from sample import psycho, utils
from sample.utils import dsp as dsp_utils


def _repeated_samples(func, key: str = "size"):
  """Decorator for sampling multiple values

  Args:
    func (callable): Function for sampling one sample
    key (str): Argument name for the number of samples

  Returns:
    Decorated function"""

  @utils.add_keyword_arg(name=key, default=None, annotation=Optional[int])
  @functools.wraps(func)
  def func_(*args, **kwargs):
    n = kwargs.pop(key, None)
    x = [func(*args, **kwargs) for _ in range(1 if n is None else n)]
    return x[0] if n is None else x

  return func_


class BeatsGenerator:
  """Random generator for audio with beats.
  It generates audio with three partials and noise,
  where two of the partials are close enough to beat.

  Args:
    f_min (float): Minimum frequency in Hertz
    f_max (float): Maximum frequency in Hertz
    f_a (float): Alpha coefficient for beta distribution of frequencies
    f_b (float): Beta coefficient for beta distribution of frequencies
    amp_min (float): Minimum amplitude value before normalization in dB
    decay_min (float): Minimum value for exponential distribution of decays
    decay (float): Expected value for exponential distribution of decays
    onlybeat (bool): If :data:`True`, then set the amplitude of the
      non-beating partial to zero
    beat_min (float): Minimum beat frequency difference in Hz
    beat_max (float): Maximum beat frequency difference in Hz
    beat_a (float): Alpha coefficient for beta distribution of
      beat frequency differences
    beat_b (float): Beta coefficient for beta distribution of
      beat frequency differences
    delta_min (float): Minimum frequency difference between beating
      and non-beating partials in Bark
    delta_max (float): Maximum frequency difference between beating
      and non-beating partials in Bark
    delta_a (float): Alpha coefficient for beta distribution of frequency
      differences between beating and non-beating partials
    delta_b (float): Beta coefficient for beta distribution of frequency
      differences between beating and non-beating partials
    snr (float): Signal-to-Noise Ratio in decibel
    rng (:class:`numpy.random.Genarator`): Random number generator instance
    seed (int): Random number generator seed"""

  def __init__(self,
               f_min: float = 200,
               f_max: float = 2000,
               f_a: float = 2,
               f_b: float = 2,
               amp_min: float = -10,
               decay_min: float = 0.5,
               decay: float = 1,
               onlybeat: bool = False,
               beat_min: float = 1.8,
               beat_max: float = 18,
               beat_a: float = 2,
               beat_b: float = 2,
               delta_min: float = 1.5,
               delta_max: float = 4,
               delta_a: float = 2,
               delta_b: float = 4,
               snr: float = 45,
               rng: Optional[np.random.Generator] = None,
               seed: Optional[int] = None):
    self.f_min = f_min
    self.f_max = f_max
    self.f_a = f_a
    self.f_b = f_b
    self.amp_min = amp_min
    self.decay_min = decay_min
    self.decay = decay
    self.onlybeat = onlybeat
    self.beat_min = beat_min
    self.beat_max = beat_max
    self.beat_a = beat_a
    self.beat_b = beat_b
    self.delta_min = delta_min
    self.delta_max = delta_max
    self.delta_a = delta_a
    self.delta_b = delta_b
    self.snr = snr
    self.rng = np.random.default_rng(seed=seed) if rng is None else rng

  @property
  def snr_amp(self) -> float:
    """Linear SNR"""
    return dsp_utils.db2a(self.snr)

  @property
  def noise_amp(self) -> float:
    """Amplitude of noise component"""
    return 1 / (1 + self.snr_amp)

  @property
  def sine_amp(self) -> float:
    """Amplitude of sinusoidal component"""
    return self.snr_amp / (1 + self.snr_amp)

  @_repeated_samples
  def beta_twosides(self,
                    a: float,
                    b: float,
                    left: float = 0,
                    right: float = 1,
                    positive: float = 0.5,
                    **kwargs):
    """Sample a value from a parametrized Beta distribution and
    randomly multiply it by either :data:`+1` or :data:`-1`

    Args:
      a (float); Alpha parameter for beta distribution
      b (float); Beta parameter for beta distribution
      left (float): Minimum value for beta distibution
      right (float): Maximum value for beta distibution
      positive (float): Probability of multiplying by :data:`+1`,
        instead of :data:`-1`
      size (int): Number of samples to draw
      kwargs: Keyword arguments for :func:`numpy.random.Genarator.beta`

    Returns:
      Random samples from the distribution"""
    x = self.rng.beta(a=a, b=b, **kwargs)
    x *= right - left
    x += left
    x *= 2.0 * (self.rng.uniform(0, 1) <= positive) - 1.0
    return x

  @_repeated_samples
  def freqs(self):
    """Sample 3 modal frequency values

    Args:
      size (int): Number of frequency triplets to draw

    Returns:
      Random frequency values"""
    beat_hz = 1 / self.beta_twosides(a=self.beat_a,
                                     b=self.beat_b,
                                     left=1 / self.beat_max,
                                     right=1 / self.beat_min)
    delta_bark = self.beta_twosides(a=self.delta_a,
                                    b=self.delta_b,
                                    left=self.delta_min,
                                    right=self.delta_max)
    b = self.beta_twosides(
        a=self.f_a,
        b=self.f_b,
        left=psycho.hz2bark(self.f_min) - min(
            (psycho.hz2bark(self.f_min) - psycho.hz2bark(self.f_min - beat_hz),
             delta_bark, 0)),
        right=psycho.hz2bark(self.f_max) - max(
            (psycho.hz2bark(self.f_max) - psycho.hz2bark(self.f_max - beat_hz),
             delta_bark, 0)),
        positive=1)
    return psycho.bark2hz((b, b, b + delta_bark)) + (0, beat_hz, 0)

  @_repeated_samples
  def decays(self):
    """Sample 3 modal decay values from an exponential distribution

    Args:
      size (int): Number of decay triplets to draw

    Returns:
      Random decay values"""
    d = self.rng.exponential(scale=self.decay - self.decay_min, size=3)
    np.add(d, self.decay_min, out=d)
    return d

  @_repeated_samples
  def amps(self):
    """Sample 3 modal amplitude values from a uniform
    distribution and normalize the sum

    Args:
      size (int): Number of amplitude triplets to draw

    Returns:
      Random amplitude values"""
    a = self.rng.uniform(self.amp_min, 0, size=3)
    dsp_utils.db2a(a, out=a)
    if self.onlybeat:
      a[-1] = 0
    np.multiply(a, self.sine_amp / np.sum(a), out=a)
    return a

  @_repeated_samples
  def noise(self, nsamples: int):
    """Sample white Gaussian noise

    Args:
      nsamples (int): Number of noise samples to draw per realization
      size (int): Number of WGN realizations to draw"""
    return self.rng.normal(scale=self.noise_amp, size=nsamples)

  @_repeated_samples
  def audio(self, dur: Optional[float] = None, fs: Optional[float] = 44100):
    """Generate random audio with beats

    Args:
      dur (float): Audio duration in seconds
      fs (float); Audio sample rate
      size (int): Number of audio signals to generate

    Return:
      Audio array, sample rate, and the tuple of frequency,
      decay, amplitude, and phase parameters"""
    if dur is None:
      dur = self.decay * 3
    freqs = self.freqs()
    decays = self.decays()
    amps = self.amps()
    phases = self.rng.uniform(0, 2 * np.pi, size=3)
    n = int(dur * fs)
    x = sample.sample.additive_synth(np.arange(n) / fs,
                                     freqs=freqs,
                                     decays=decays,
                                     amps=amps,
                                     phases=phases)
    x += self.noise(nsamples=n)
    return x, fs, (freqs, decays, amps, phases)
