"""Signal processing functions for SMS"""
import numpy as np
from scipy import fft


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
