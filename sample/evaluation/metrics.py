"""Metrics for model evaluation"""
import functools
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from sample import psycho
from sample.utils import dsp as dsp_utils
from scipy import signal


def lp_distance(x, y, p: float = 1):
  """Compute the distance between two vectors as
  the lp-norm of their difference

  Args:
    x (array): First vector
    y (array): Second vector
    p (float): Exponent for the lp-norm

  Returns:
    float: The distance"""
  tmp = np.empty_like(x)
  np.subtract(x, y, out=tmp)
  np.abs(tmp, out=tmp)
  if p != 1:
    np.power(tmp, p, out=tmp)
  d = np.sum(tmp)
  if p != 1:
    d = np.power(d, 1 / p)
  return d


def lin_log_spectral_loss(x,
                          y,
                          n: int = 2048,
                          olap: float = 0.75,
                          w: Optional[np.ndarray] = None,
                          wtype: str = "hamming",
                          wsize: Optional[int] = None,
                          alpha: Optional[float] = None,
                          norm_p: float = 1.0,
                          floor_db: float = -60,
                          **kwargs):
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
    kwargs: Keyword arguments for :func:`scipy.signal.stft`

  Return:
    float: loss value"""
  if w is None:
    if wsize is None:
      wsize = n
    w = signal.get_window(window=wtype, Nx=wsize)
  if alpha is None:
    alpha = 1 / max(1, -floor_db)

  noverlap = int(olap * np.size(w))
  _, _, x_stft = signal.stft(x,
                             window=w,
                             nperseg=np.size(w),
                             nfft=n,
                             noverlap=noverlap,
                             **kwargs)
  _, _, y_stft = signal.stft(y,
                             window=w,
                             nperseg=np.size(w),
                             nfft=n,
                             noverlap=noverlap,
                             **kwargs)
  x_stft = np.abs(x_stft)
  y_stft = np.abs(y_stft)
  loss = lp_distance(x_stft, y_stft, p=norm_p)
  dsp_utils.a2db(x_stft, out=x_stft, floor=floor_db, floor_db=True)
  dsp_utils.a2db(y_stft, out=y_stft, floor=floor_db, floor_db=True)
  loss += alpha * lp_distance(x_stft, y_stft, p=norm_p)
  return loss


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


class CochleagramLoss:
  """Class for computing losses on cochleagrams (lower is better)

  Args:
    fs (float): Sample frequency
    postprocessing (callable): If not :data:`None`, then apply this function
      to the cochleagram matrix. Default is :func:`hwr`, if the cochleagram
      is real, otherwise it is :func:`numpy.abs`
    method (str): Convolution method (either :data:`"auto"`,
      :data:`"fft"`, :data:`"direct"`, or :data:`"overlap-add"`)
    stride (int): Time-step for output signal.
      Can't be used in conjunction with :data:`method`
    analytical (str): Compute the analytical signal of the cochleagram:

      - if :data:`"input"`, then compute the analytical signal
        of the input (fast, accurate in the middle, bad boundary conditions)
      - if :data:`"ir"` (suggested), then compute the analytical signal
        of the IRs (fast, tends to underestimate amplitude,
        good boundary conditions)
      - if :data:`"output"`, then compute the analytical signal
        of the output (slowest, most accurate)
    p (float): Exponent for the lp-norm
    **kwargs: Keyword arguments for
      :class:`sample.psycho.GammatoneFilterbank`"""

  def __init__(self,
               fs: float,
               analytical: Optional[str] = "ir",
               method: Optional[str] = None,
               stride: Optional[int] = None,
               p: float = 1,
               **kwargs):
    kpp = "postprocessing"
    if kpp in kwargs:
      self.postprocessing = {kpp: kwargs.pop(kpp)}
    elif analytical is not None:
      self.postprocessing = {kpp: np.abs}
    else:
      self.postprocessing = {}
    self.filterbank = psycho.GammatoneFilterbank(**kwargs).precompute(
        fs=fs, analytical=analytical == "ir")
    self.analytical = analytical
    self.method = method
    self.stride = stride
    self.p = p

  def cochleagram(self, x: np.ndarray) -> np.ndarray:
    """Compute cochleagram for one input

    Args:
      x (array): Input signal

    Returns:
      array: Cochleagram"""
    return psycho.cochleagram(x,
                              filterbank=self.filterbank,
                              analytical=self.analytical,
                              method=self.method,
                              stride=self.stride,
                              **self.postprocessing)[0]

  def lp_distance(self, x: np.ndarray, y: np.ndarray):
    """Compute the distance between two vectors as
    the lp-norm of their difference

    Args:
      x (array): First vector
      y (array): Second vector

    Returns:
      float: The distance"""
    return lp_distance(x=x, y=y, p=self.p)

  def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
    """Compute cochleagram loss

    Args:
      x (array): First vector
      y (array): Second vector

    Returns:
      float: Loss value"""
    return self.lp_distance(self.cochleagram(x), self.cochleagram(y))
