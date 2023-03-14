"""Automatic optimization of SAMPLE hyperparameters"""
import collections
import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.optimize
import skopt
import tqdm
from scipy import signal
from sklearn import base

import sample
import sample.utils
import sample.utils.learn
from sample.evaluation import metrics

utils = sample.utils

# --- Monkey patch to solve issue ---------------------------------------------
#   https://github.com/scikit-optimize/scikit-optimize/issues/1138
np.int = int
# -----------------------------------------------------------------------------


@utils.deprecated_argument("sinusoidal_model__log_n", "sinusoidal__log_n")
@utils.deprecated_argument("sinusoidal_model__wtype", "sinusoidal__wtype")
@utils.deprecated_argument("sinusoidal_model__wsize", "sinusoidal__wsize")
@utils.deprecated_argument("sinusoidal_model__overlap", "sinusoidal__overlap")
def sample_kwargs_remapper(sinusoidal__log_n: Optional[int] = None,
                           sinusoidal__wtype: str = "hamming",
                           sinusoidal__wsize: float = 1.0,
                           sinusoidal__overlap: float = 0.5,
                           **kwargs) -> Dict[str, Any]:
  """Default argument remapper for :class:`SAMPLEOptimizer`. It remaps stft
  window paramaters and lets every other parameter pass through

  Args:
    sinusoidal__log_n (int): Logarithm of fft size. Will be remapped to
      :data:`sinusoidal__n` if not in :data:`kwargs`
    sinusoidal__wtype (str): Name of the window to use. Default is
      :data:`"hamming"`. It is used to compute the window
      :data:`sinusoidal__w` if not in :data:`kwargs`
    sinusoidal__wsize (float): Window size as a fraction of fft size.
      Default is :data:`1.0`. It is used to compute the window
      :data:`sinusoidal__w` if not in :data:`kwargs`
    sinusoidal__overlap (float): Window overlap as a fraction of the
      window size. Default is :data:`0.5`. It is used to compute the hop size
      :data:`sinusoidal__tracker__h` if not in :data:`kwargs`
    **kwargs: Pass-through keyword arguments

  Returns:
    dict: Remapped keyword arguments"""
  # FFT size from log-size
  if sinusoidal__log_n is not None and \
    "sinusoidal__n" not in kwargs:
    kwargs["sinusoidal__n"] = 1 << sinusoidal__log_n
  # Window from name and size
  if "sinusoidal__w" not in kwargs:
    wsize = int(sample.SAMPLE(**kwargs).sinusoidal.n * sinusoidal__wsize)
    kwargs["sinusoidal__w"] = signal.get_window(window=sinusoidal__wtype,
                                                Nx=wsize)
  # Hop-size from overlap and window size
  if "sinusoidal__tracker__h" not in kwargs:
    wsize = np.size(sample.SAMPLE(**kwargs).sinusoidal.w)
    hopsize = int((1 - sinusoidal__overlap) * wsize)
    hopsize = max(min(hopsize, wsize), 1)
    kwargs["sinusoidal__tracker__h"] = hopsize
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

  @utils.deprecated_argument(
      "sample_fn",
      convert=lambda _, **kwargs:
      ("model", kwargs["sample_fn"]
       (**kwargs.get("remap", sample_kwargs_remapper)({} if kwargs.get(
           "sample_kw", None) is None else kwargs.get("sample_kw", None)))),
      msg="Provide a sample.SAMPLE instance with the desired parameters, "
      "instead of constructor and arguments")
  @utils.deprecated_argument(
      "sample_kw",
      convert=lambda _, **kwargs:
      ("model", kwargs.get("model", sample.SAMPLE()).set_params(**kwargs.get(
          "remap", sample_kwargs_remapper)(**kwargs["sample_kw"]))),
      msg="Provide a sample.SAMPLE instance with the desired parameters, "
      "instead of constructor and arguments")
  def __init__(self,
               model: sample.SAMPLE = None,
               loss_fn: Callable[[np.ndarray, np.ndarray],
                                 float] = metrics.multiscale_spectral_loss,
               loss_kw: Optional[Dict[str, Any]] = None,
               remap: Optional[Callable[...,
                                        Dict[str,
                                             Any]]] = sample_kwargs_remapper,
               clip: bool = True,
               **kwargs):
    self.model = model
    self.loss_fn = functools.partial(loss_fn, **utils.default_kws(loss_kw))
    self.dimensions = collections.OrderedDict(kwargs)
    self.remap = remap
    self.clip = clip

  @utils.learn.default_property
  def model(self):
    return sample.SAMPLE()

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

    def loss_(args: Tuple = (), x=x, sinusoidal__tracker__fs=fs,
              **kwargs) -> float:
      model = base.clone(self.model)
      model.set_params(**self._kwargs(
          *args, **kwargs, sinusoidal__tracker__fs=sinusoidal__tracker__fs))
      model.fit(x)
      y = model.predict(np.arange(x.size) / sinusoidal__tracker__fs,
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
      ignore_warnings: bool = True,
      fit_kws: Optional[Dict[str, Any]] = None,
      **kwargs) -> Tuple[sample.SAMPLE, scipy.optimize.OptimizeResult]:
    """Use :func:`skopt.gp_minimize` to tune the hyperparameters

    Args:
      x (array): Audio samples
      fs (float): Sample rate
      ignore_warnings (bool): If :data:`True` (default), then ignore warnings
        while optimizing
      fit_kws (dict): Arguments for the :func:`self.model.fit` method
      **kwargs: Keyword arguments for :func:`skopt.gp_minimize`

    Returns:
      SAMPLE, OptimizeResult: Best model, and optimization summary"""
    if state is not None and "x0" not in kwargs and "y0" not in kwargs:
      kwargs["x0"] = state.x_iters
      kwargs["y0"] = state.func_vals
    with warnings.catch_warnings():
      if ignore_warnings:
        warnings.simplefilter("ignore", Warning)
      res = skopt.gp_minimize(self.loss(x, fs), self.dimensions.values(),
                              **kwargs)
    model = base.clone(self.model)
    model.set_params(**self._kwargs(*res.x, sinusoidal__tracker__fs=fs))
    model.fit(x, **utils.default_kws(fit_kws))
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
