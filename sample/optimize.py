"""Automatic optimization of SAMPLE hyperparameters"""
import collections
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.optimize
import skopt
import tqdm
from scipy import signal

import sample
from sample.evaluation import metrics


def sample_kwargs_remapper(sinusoidal_model__log_n: Optional[int] = None,
                           sinusoidal_model__wtype: str = "hamming",
                           sinusoidal_model__wsize: float = 1.0,
                           sinusoidal_model__overlap: float = 0.5,
                           **kwargs) -> Dict[str, Any]:
  """Default argument remapper for :class:`SAMPLEOptimizer`. It remaps stft
  window paramaters and lets every other parameter pass through

  Args:
    sinusoidal_model__log_n (int): Logarithm of fft size. Will be remapped to
      :data:`sinusoidal_model__n` if not in :data:`kwargs`
    sinusoidal_model__wtype (str): Name of the window to use. Default is
      :data:`"hamming"`. It is used to compute the window
      :data:`sinusoidal_model__w` if not in :data:`kwargs`
    sinusoidal_model__wsize (float): Window size as a fraction of fft size.
      Default is :data:`1.0`. It is used to compute the window
      :data:`sinusoidal_model__w` if not in :data:`kwargs`
    sinusoidal_model__overlap (float): Window overlap as a fraction of the
      window size. Default is :data:`0.5`. It is used to compute the hop size
      :data:`sinusoidal_model__h` if not in :data:`kwargs`
    **kwargs: Pass-through keyword arguments

  Returns:
    dict: Remapped keyword arguments"""
  # FFT size from log-size
  if sinusoidal_model__log_n is not None and \
    "sinusoidal_model__n" not in kwargs:
    kwargs["sinusoidal_model__n"] = 1 << sinusoidal_model__log_n
  # Window from name and size
  if "sinusoidal_model__w" not in kwargs:
    wsize = int(
        sample.SAMPLE(**kwargs).get_params()["sinusoidal_model__n"] *
        sinusoidal_model__wsize)
    kwargs["sinusoidal_model__w"] = signal.get_window(
        window=sinusoidal_model__wtype, Nx=wsize)
  # Hop-size from overlap and window size
  if "sinusoidal_model__h" not in kwargs:
    wsize = np.size(sample.SAMPLE(**kwargs).get_params()["sinusoidal_model__w"])
    hopsize = int((1 - sinusoidal_model__overlap) * wsize)
    hopsize = max(min(hopsize, wsize), 1)
    kwargs["sinusoidal_model__h"] = hopsize
  return kwargs


class SAMPLEOptimizer:
  """Hyperparameter optimizer for a SAMPLE model,
  based on Gaussian Process minimization

  Args:
    sample_fn (callable): Constructor for a SAMPLE model
    sample_kw (dict): Keyword arguments for :data:`sample_fn`.
      These parameters will not be optimized. These parameters will
      potentially be remapped by :data:`remap`
    loss_fn (callable): Loss function. It should take, as two positional
      arguments, the arrays of original and resynthesised audio samples
    loss_kw (dict): Keyword arguments for the loss function
    remap (callable): Function that accepts keyword arguments and
      returns a new dictionary of keyword arguments for :data:`sample_fn`.
      Default is :func:`sample_kwargs_remapper`
    clip (bool): If :data:`True` (default), then clip resynthesised audio
      to the same peak of the original audio
    **kwargs: Parameters to optimize. See :func:`skopt.gp_minimize`
      **dimensions** for definition options"""

  def __init__(self,
               sample_fn: Callable[..., sample.SAMPLE] = sample.SAMPLE,
               sample_kw: Optional[Dict[str, Any]] = None,
               loss_fn: Callable[[np.ndarray, np.ndarray],
                                 float] = metrics.multiscale_spectral_loss,
               loss_kw: Optional[Dict[str, Any]] = None,
               remap: Optional[Callable[...,
                                        Dict[str,
                                             Any]]] = sample_kwargs_remapper,
               clip: bool = True,
               **kwargs):
    self.loss_fn = loss_fn if loss_kw is None else functools.partial(
        loss_fn, **loss_kw)
    self.sample_fn = sample_fn
    self.sample_kw = {} if sample_kw is None else sample_kw
    self.dimensions = collections.OrderedDict(kwargs)
    self.remap = remap
    self.clip = clip

  def _kwargs(self, *args, **kwargs) -> Dict[str, Any]:
    """Compose positional and keyword arguments together.
    Positional arguments' kewyords are the keys of the :attr:`dimensions`

    Args:
      *args: Positional arguments, their number and order must match the number
        and order of the :attr:`dimensions`
      **kwargs: Additional keyword arguments

    Returns:
      dict: Composed keyword arguments"""
    if len(args) != len(self.dimensions):
      raise ValueError(f"Expected {len(self.dimensions)} "
                       f"positional arguments, got {len(args)}")
    if self.remap is not None:
      kwargs = self.remap(**dict(zip(self.dimensions, args)), **kwargs)
    return kwargs

  def loss(self, x: np.ndarray, fs: float) -> Callable[..., float]:
    """Define a loss function for the target audio based on
    computing :attr:`loss_fn` on the target and resynthesised audio

    Args:
      x (array): Audio samples
      fs (float): Sample rate

    Returns:
      callable: Loss function"""
    if self.clip:
      peak = np.max(np.abs(x))

    def loss_(args: Tuple = (), x=x, sinusoidal_model__fs=fs,
              **kwargs) -> float:
      model = self.sample_fn(
          **self._kwargs(*args,
                         **kwargs,
                         **self.sample_kw,
                         sinusoidal_model__fs=sinusoidal_model__fs))
      model.fit(x)
      y = model.predict(np.arange(x.size) / sinusoidal_model__fs,
                        phases="random")
      if self.clip:
        np.clip(y, -peak, peak, out=y)
      return self.loss_fn(x, y)

    return loss_

  def gp_minimize(
      self,
      x: np.ndarray,
      fs: float = 44100,
      state: Optional[scipy.optimize.OptimizeResult] = None,
      **kwargs) -> Tuple[sample.SAMPLE, scipy.optimize.OptimizeResult]:
    """Use :func:`skopt.gp_minimize` to tune the hyperparameters

    Args:
      x (array): Audio samples
      fs (float): Sample rate
      **kwargs: Keyword arguments for :func:`skopt.gp_minimize`

    Returns:
      SAMPLE, OptimizeResult: Best model, and optimization summary"""
    if state is not None and "x0" not in kwargs and "y0" not in kwargs:
      kwargs["x0"] = state.x_iters
      kwargs["y0"] = state.func_vals
    res = skopt.gp_minimize(self.loss(x, fs), self.dimensions.values(),
                            **kwargs)
    model = self.sample_fn(
        **self._kwargs(*res.x, sinusoidal_model__fs=fs, **self.sample_kw))
    model.fit(x)
    return model, res


class TqdmCallback:
  """Callback for using tqdm with :class:`SAMPLEOptimizer`

  Args:
    sample_opt (SAMPLEOptimizer): Optimizer instance
    n_calls (int): Number of total calls
    n_initial_points (int): Number of initial (random) points
    tqdm_fn (callable): Constructor for a tqdm object
    minimum (bool): If :data:`True` (default), show current
      minimum in postfix"""

  def __init__(self,
               sample_opt: SAMPLEOptimizer,
               n_calls: int,
               n_initial_points: int = 0,
               tqdm_fn: Callable[..., tqdm.tqdm] = tqdm.tqdm,
               minimum: bool = True):
    self.sample_opt = sample_opt
    self.n_calls = n_calls
    self.n_initial_points = n_initial_points
    self.tqdm_fn = tqdm_fn
    self.i = None
    self.tqdm = None
    self.minimum = minimum

  @property
  def started(self) -> bool:
    """If :data:`True`, the callback has already been started"""
    return self.tqdm is not None or self.i is not None

  def reset(self) -> "TqdmCallback":
    """Reset the state of the callback, e.g. for using it again"""
    if self.tqdm is not None:
      self.tqdm.close()
    self.i = None
    self.tqdm = None
    return self

  def start(self) -> "TqdmCallback":
    """Start the callback.
    Calls to this method initialize internal objects"""
    self.tqdm = self.tqdm_fn(total=self.n_calls)
    self.i = 0
    return self

  def __call__(self, res: scipy.optimize.OptimizeResult):
    """Callback function

    Args:
      res (OptimizeResult): Current result state"""
    if not self.started:
      self.start()
    self.i += 1
    self.tqdm.update()
    if self.i <= self.n_initial_points:
      postfix_str = "stage=randomize"
    else:
      postfix_str = "stage=minimize"
    if self.minimum:
      params_str = ", ".join(
          f"{k}={v}" for k, v in zip(self.sample_opt.dimensions, res.x))
      postfix_str += f", minimum=(value={res.fun}, params=({params_str}))"
    if len(postfix_str) > 0:
      self.tqdm.set_postfix_str(postfix_str)
