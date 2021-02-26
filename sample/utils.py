"""Utilities for tests and notebooks"""
import numpy as np


def test_audio(
  f: np.ndarray = np.array([440, 650, 690]),
  a: np.ndarray = np.array([1, .5, .45]),
  d: np.ndarray = np.array([.66, .4, .35]),
  dur: float = 2,
  fs: int = 44100,
  noise_db: float = -60,
  seed: int = 42,
):
  """Synthesize a modal-like sound for test purposes

  Args:
    f (array): Modal frequencies
    a (array): Modal amplitudes
    d (array): Modal decays
    dur (float): Duration in seconds
    fs (int): Sampling frequency in Hz
    noise_db (float): Gaussian noise magnitude in dB
    seed (int): Gaussian noise seed

  Returns:
    array: Array of audio samples"""
  t = np.linspace(0, dur, int(dur * fs), endpoint=False)
  x = np.squeeze(np.reshape(a, (1, -1)) @ (
    np.exp(np.reshape(-2 / d, (-1, 1)) * np.reshape(t, (1, -1))) *
    np.sin(np.reshape(f, (-1, 1)) * 2 * np.pi * np.reshape(t, (1, -1)))
  ))
  x = x / np.max(np.abs(x))

  np.random.seed(seed)
  x = x + 10**(noise_db/20) * np.random.randn(*x.shape)
  x = x / np.max(np.abs(x))
  return x
