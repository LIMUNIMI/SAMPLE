"""Automatic optimization of SAMPLE hyperparameters"""
from argparse import ArgumentError
import collections
import functools
from scipy import signal
from typing import Any, Callable, Dict, Optional, Tuple

import scipy.optimize

import sample
from sample.evaluation import metrics
import numpy as np
import skopt


def sample_kwargs_remapper(sinusoidal_model__log_n: Optional[int] = None,
                           sinusoidal_model__wtype: str = "hamming",
                           sinusoidal_model__wsize: float = 1.0,
                           sinusoidal_model__overlap: float = 0.5,
                           **kwargs) -> Dict[str, Any]:
  # FFT size from log-size
  if sinusoidal_model__log_n is not None and "sinusoidal_model__n" not in kwargs:
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

  def __init__(self,
               sample_fn: Callable[..., sample.SAMPLE] = sample.SAMPLE,
               sample_kw: Optional[Dict[str, Any]] = None,
               loss_fn: Callable[..., float] = metrics.multiscale_spectral_loss,
               loss_kw: Optional[Dict[str, Any]] = None,
               remap: Optional[Callable[[Dict[str, Any]],
                                        Dict[str,
                                             Any]]] = sample_kwargs_remapper,
               **kwargs):
    self.loss_fn = loss_fn if loss_kw is None else functools.partial(
        loss_fn, **loss_kw)
    self.sample_fn = sample_fn
    self.sample_kw = {} if sample_kw is None else sample_kw
    self.dimensions = collections.OrderedDict(kwargs)
    self.remap = remap

  def _kwargs(self, *args, **kwargs):
    if len(args) != len(self.dimensions):
      raise ArgumentError(
          None, f"Expected {len(self.dimensions)} "
          f"positional arguments, got {len(args)}")
    if self.remap is not None:
      kwargs = self.remap(**dict(zip(self.dimensions, args)), **kwargs)
    return kwargs

  def loss(self, x: np.ndarray, fs: float) -> Callable[..., float]:

    def loss_(args: Tuple = (),
              *more_args,
              x=x,
              sinusoidal_model__fs=fs,
              **kwargs) -> float:
      model = self.sample_fn(
          **self._kwargs(*args,
                         *more_args,
                         **kwargs,
                         **self.sample_kw,
                         sinusoidal_model__fs=sinusoidal_model__fs))
      model.fit(x)
      y = model.predict(np.arange(x.size) / sinusoidal_model__fs)
      return self.loss_fn(x, y)

    return loss_

  def gp_minimize(
      self,
      x: np.ndarray,
      fs: float = 44100,
      **kwargs) -> Tuple[sample.SAMPLE, scipy.optimize.OptimizeResult]:
    res = skopt.gp_minimize(self.loss(x, fs), self.dimensions.values(),
                            **kwargs)
    model = self.sample_fn(
        **self._kwargs(*res.x, sinusoidal_model__fs=fs, **self.sample_kw))
    model.fit(x)
    return model, res
