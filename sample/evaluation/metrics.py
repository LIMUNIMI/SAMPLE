"""Metrics for model evaluation"""
import copy
import functools
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from sample.sms import sm
from scipy import signal


class _STFTDifferenceCalculator(sm.SinusoidalModel):
  """Class for computing losses on the STFT online

  Args:
    w: Analysis window. Defaults to None (if None,
      the :attr:`default_window` is used)
    n (int): FFT size. Defaults to 2048
    h (int): Window hop size. Defaults to 500
    alpha (float): Weight of the log-difference
    norm_p (float): Exponent patameter for norm. Default is :data:`1.0`
    floor_db (float): Minimum magnitude for STFT in dB"""

  def __init__(
      self,
      w: Optional[np.ndarray] = None,
      n: int = 2048,
      h: int = 500,
      alpha: float = 1.0,
      norm_p: float = 1.0,
      floor_db: float = -60,
  ):
    super().__init__(
        w=w,
        n=n,
        h=h,
    )
    self.alpha = alpha
    self.norm_p = norm_p
    self.floor_db = floor_db

  def _lpp_distance(self, x: np.ndarray, y: np.ndarray) -> float:
    """Compute the p-norm distance between x and y to the power p

    Args:
      x (array): first input
      y (array): second input

    Returns:
      float: p-norm distance to the power p"""
    return np.sum(np.power(np.abs(x - y), self.norm_p))

  def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
    """Compute the STFT differences online

    Args:
      x (array): first audio input
      y (ignored): second audio input
      kwargs: Any parameter, overrides initialization

    Returns:
      SinusoidalModel: self"""
    self.set_params(**kwargs)
    self.w_ = self.normalized_window

    x_dft = (f[0] for f in copy.deepcopy(self).dft_frames(x))
    y_dft = (f[0] for f in copy.deepcopy(self).dft_frames(y))

    diff = np.zeros((2, 1))
    for x_fft, y_fft in zip(x_dft, y_dft):  # these are log magnitudes
      diff[0, 0] += self._lpp_distance(np.power(10, x_fft / 20),
                                       np.power(10, y_fft / 20))  # lin diff
      diff[1, 0] += self._lpp_distance(np.maximum(x_fft, self.floor_db),
                                       np.maximum(y_fft,
                                                  self.floor_db))  # log diff
    self.loss_ = np.squeeze(
        np.array([[1, self.alpha]]) @ np.power(diff, 1 / self.norm_p))[()]
    return self


def lin_log_spectral_loss(
    x,
    y,
    n: int = 2048,
    olap: float = 0.75,
    w: Optional[np.ndarray] = None,
    wtype: str = "hamming",
    wsize: Optional[int] = None,
    alpha: Optional[float] = None,
    norm_p: float = 1.0,
    floor_db: float = -60,
):
  """Compute a sum of linear and log loss on the STFT (lower is better)

  Args:
    x (array): First audio input
    y (array): Second audio input
    w: Analysis window. Defaults to None (if None,
      the :attr:`default_window` is used)
    n (int): FFT size. Defaults to 2048
    olap (float): Window overlap, as a fraction of the window size
    alpha (float): Weight of the log-difference
    norm_p (float): Exponent patameter for norm. Default is :data:`1.0`
    floor_db (float): Minimum magnitude for STFT in dB

  Return:
    float: loss value"""
  if w is None:
    if wsize is None:
      wsize = n
    w = signal.get_window(window=wtype, Nx=wsize)
  if alpha is None:
    alpha = 1 / max(1, -floor_db)
  return _STFTDifferenceCalculator(
      n=n,
      h=max(1, int(np.size(w) * (1 - olap))),
      w=w,
      alpha=alpha,
      norm_p=norm_p,
      floor_db=floor_db,
  ).fit(x, y).loss_


class MultiScaleSpectralLoss:
  """Class for computing multiscale losses on the STFT online
  and in parallel

  Args:
    spectral_loss (callable): Base function for computing a spectral loss
    stfts (iterable of dict): Multiple dictionaries of
      keyword arguments for :func:`spectral_loss`"""

  def __init__(self, spectral_loss: Callable, stfts: Iterable[Dict[str, Any]]):
    self.funcs = [functools.partial(spectral_loss, **kw) for kw in stfts]

  def __call__(self,
               *args,
               pool: Optional[mp.Pool] = None,
               njobs: Optional[int] = None,
               **kwargs) -> float:
    """Compute the loss

    Args:
      args: Additional positional arguments for the base loss function
      pool (multiprocessing.Pool): If not :data:`None`, then compute the
        losses in parallel processes from this pool
      njobs (int): If not :data:`None`, then compute the
        losses in parallel using a process pool with :data:`njobs` workers
      kwargs: Additional keyword arguments for the base loss function

    Returns:
      float: Sum of all losses"""
    if pool is None:
      if njobs is None:
        it = (f(*args, **kwargs) for f in self.funcs)
      else:
        with mp.Pool(processes=njobs) as pool_:
          return self(*args, pool=pool_, njobs=njobs, **kwargs)
    else:
      results = [pool.apply_async(f, args, kwargs) for f in self.funcs]
      it = (r.get() for r in results)
    return sum(it)


def multiscale_spectral_loss(x,
                             y,
                             *args,
                             spectral_loss: Callable = lin_log_spectral_loss,
                             stfts: Iterable[Dict[str, Any]] = tuple(
                                 dict(n=1 << i) for i in range(6, 12)),
                             **kwargs) -> float:
  """Compute a multiscale spectral loss

  Args:
    x (array): First audio input
    y (array): Second audio input
    args: Additional positional arguments for the base loss function
    spectral_loss (callable): Base function for computing a spectral loss.
      Default is :func:`lin_log_spectral_loss`
    stfts (iterable of dict): Multiple dictionaries of
    pool (multiprocessing.Pool): If not :data:`None`, then compute the
      losses in parallel processes from this pool
    njobs (int): If not :data:`None`, then compute the
      losses in parallel using a process pool with :data:`njobs` workers
    kwargs: Additional keyword arguments for the base loss function

  Return:
    float: Sum of loss values"""
  return MultiScaleSpectralLoss(stfts=stfts,
                                spectral_loss=spectral_loss)(x, y, *args,
                                                             **kwargs)
