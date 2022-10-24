"""Signal processing functions for SMS"""
import functools
from typing import Optional, Tuple, Union

import numpy as np
from sample.utils import dsp as dsp_utils
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
  x_z[:w.size - hw] = xw[hw:]
  x_z[-hw:] = xw[:hw]

  x_fft = fft.rfft(x_z)
  ax = np.abs(x_fft)
  ax = np.maximum(ax, np.finfo(float).eps, out=ax)  # avoid zeros in log
  mx = dsp_utils.a2db.floorless(ax)

  x_fft[ax < tol] = complex(0)  # pylint: disable=E1137
  px = np.unwrap(np.angle(x_fft))
  return mx, px


def peak_detect(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
  """Detect peaks (local maxima) in a signal

  Arguments:
    x (array): Input signal
    t (float): Threshold (optional)

  Returns:
    array: The indices of the peaks in x"""
  conditions = [
      np.greater(x[1:-1], x[2:]),  # greater than next sample
      np.greater(x[1:-1], x[:-2]),  # greater than previous sample
  ]
  if t is not None:
    conditions.append(np.greater(x[1:-1], t))  # above threshold
  return np.flatnonzero(functools.reduce(
      np.logical_and, conditions)) + 1  # compensate for skipping first sample


def peak_refine(
    ploc: np.ndarray,
    mx: np.ndarray,
    px: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray,
                                                np.ndarray]]:
  """Refine detected peaks with parabolic approximation

  Arguments:
    ploc (array): Peak locations
    mx (array): Magnitude spectrum in dB
    px (array): Phase spectrum

  Returns:
    (array, array, array): Interpolated peak locations, magnitudes and phases"""
  mx_l = mx[ploc - 1]
  mx_c = mx[ploc]
  mx_r = mx[ploc + 1]

  dmx = mx_l - mx_r
  ploc_d = .5 * dmx / (mx_l - 2 * mx_c + mx_r)
  ploc_i = ploc + ploc_d  # x-coordinate of vertex
  pmag_i = mx_c - .25 * dmx * ploc_d  # y-coordinate of vertex
  if px is None:
    return ploc_i, pmag_i
  pph_i = np.interp(  # linear interpolation for phase
      ploc_i, np.arange(0, px.size), px)
  return ploc_i, pmag_i, pph_i


def peak_detect_interp(
    mx: np.ndarray,
    px: Optional[np.ndarray] = None,
    t: Optional[float] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray,
                                                np.ndarray]]:
  """Detect peaks (local maxima) in a signal, refining the value with
  parabolic interpolation

  Arguments:
    mx (array): Magnitude spectrum in dB
    px (array): Phase spectrum
    t (float): Threshold (optional)

  Returns:
    (array, array, array): Interpolated peak locations, magnitudes and phases"""
  return peak_refine(peak_detect(mx, t=t), mx, px)
