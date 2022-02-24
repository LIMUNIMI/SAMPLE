"""Random generation of modal-like sounds"""
import functools
import inspect
from typing import Optional

import numpy as np
import sample.sample
from sample import psycho
from scipy import special


class BeatsGenerator:
  """Random generator for audio with beats.
  It generates audio with three partials and noise,
  where two of the partials are close enough to beat.

  Args:
    f_min (float): Minimum frequency in Hertz
    f_max (float): Maximum frequency in Hertz
    f_a (float): Alpha coefficient for beta distribution of frequencies
    f_b (float): Beta coefficient for beta distribution of frequencies
    decay (float): Expected value for exponential distribution of decays
    beat_min (float): Minimum beat frequency difference in Bark
    beat_max (float): Maximum beat frequency difference in Bark
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
               f_min: float = 80,
               f_max: float = 12000,
               f_a: float = 2,
               f_b: float = 2,
               decay: float = 1,
               beat_min: float = 0.05,
               beat_max: float = 0.5,
               beat_a: float = 2,
               beat_b: float = 4,
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
    self.decay = decay
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
    return np.power(10, self.snr / 20)

  @property
  def noise_amp(self) -> float:
    """Amplitude of noise component"""
    return 1 / (1 + self.snr_amp)

  @property
  def sine_amp(self) -> float:
    """Amplitude of sinusoidal component"""
    return self.snr_amp / (1 + self.snr_amp)

  @staticmethod
  def _repeated_samples(func, key: str = "size"):
    """Decorator for sampling multiple values

    Args:
      func (callable): Function for sampling one sample
      key (str): Argument name for the number of samples

    Returns:
      Decorated function"""

    @functools.wraps(func)
    def func_(*args, **kwargs):
      n = kwargs.pop(key, 1)
      x = [func(*args, **kwargs) for _ in range(n)]
      if n == 1:
        return x[0]
      else:
        return x

    # Add parameter to signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    params.insert(
        len(params) - (params[-1].kind == inspect.Parameter.VAR_KEYWORD),
        inspect.Parameter(name=key,
                          kind=inspect.Parameter.KEYWORD_ONLY,
                          default=1,
                          annotation=int))
    func_.__signature__ = sig.replace(parameters=params)

    return func_

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
    beat = self.beta_twosides(a=self.beat_a,
                              b=self.beat_b,
                              left=self.beat_min,
                              right=self.beat_max)
    delta = self.beta_twosides(a=self.delta_a,
                               b=self.delta_b,
                               left=self.delta_min,
                               right=self.delta_max)
    bark = self.beta_twosides(a=self.f_a,
                              b=self.f_b,
                              left=psycho.hz2bark(self.f_min) - min(
                                  (beat, delta, 0)),
                              right=psycho.hz2bark(self.f_max) - max(
                                  (beat, delta, 0)),
                              positive=1)
    return psycho.bark2hz(np.array((bark, bark + beat, bark + delta)))

  @_repeated_samples
  def decays(self):
    """Sample 3 modal decay values from an exponential distribution

    Args:
      size (int): Number of decay triplets to draw

    Returns:
      Random decay values"""
    return self.rng.exponential(scale=self.decay, size=3)

  @_repeated_samples
  def amps(self):
    """Sample 3 modal amplitude values from an uniform distribution
    and correct the sum with a softmax function

    Args:
      size (int): Number of amplitude triplets to draw

    Returns:
      Random amplitude values"""
    return special.softmax(self.rng.uniform(0, 1, size=3)) * self.sine_amp

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
      decay, and amplitude parameters"""
    if dur is None:
      dur = self.decay * 3
    freqs = self.freqs()
    decays = self.decays()
    amps = self.amps()
    n = int(dur * fs)
    x = sample.sample.additive_synth(np.arange(n) / fs,
                                     freqs=freqs,
                                     decays=decays,
                                     amps=amps)
    x += self.noise(nsamples=n)
    return x, fs, (freqs, decays, amps)
