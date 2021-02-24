"""Signal processing functions for SMS"""
import numpy as np
from scipy import fft
import functools
from typing import Optional


def dft(
  x: np.ndarray,
  w: np.ndarray,
  n: int,
  tol: float = 1e-14,
):
  """Discrete Fourier Transform with zero-phase windowing

  Args:
    x (array): Input
    w (array): Window
    n (int): FFT size
    tol (float): Threshold below which all values are set to 0
      for phase computation. Defaults to 1e-14

  Returns:
    (array, array): Magnitude (in dB) and (unwrapped) phase spectra"""
  hw = w.size // 2

  xw = x * w
  x_z = np.zeros(n)
  x_z[:w.size-hw] = xw[hw:]
  x_z[-hw:] = xw[:hw]

  x_fft = fft.rfft(x_z)
  ax = np.maximum(np.abs(x_fft), np.finfo(float).eps)  # avoid zeros in log
  mx = 20 * np.log10(ax)

  x_fft[ax < tol] = complex(0)
  px = np.unwrap(np.angle(x_fft))
  return mx, px


def peak_detect(x: np.ndarray, t: Optional[float] = None):
  """Detect peaks (local maxima) in a signal

  Arguments:
    x (array): Input signal
    t (float): Threshold (optional)

  Returns:
    array: The indices of the peaks in x"""
  conditions = [
    np.greater(x[1:-1], x[2:]),   # greater than next sample
    np.greater(x[1:-1], x[:-2]),  # greater than previous sample
  ]
  if t is not None:
    conditions.append(np.greater(x[1:-1], t))  # above threshold
  return np.flatnonzero(
    functools.reduce(np.logical_and, conditions)
  ) + 1  # compensate for skipping first sample
