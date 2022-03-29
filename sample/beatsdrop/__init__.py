"""Implementation of Beats DROP (Beats
Duality for the Resolution Of Partials)"""
import functools
import itertools
from typing import Callable, Optional, Union, Tuple, List, Iterable

import numpy as np
import paragraph as pg
from sample import utils
from sample.utils import np2pg
from scipy import integrate

FloatOrCallable = Union[float, Callable[[np.ndarray], np.ndarray]]


def _float_or_call(
    foc: FloatOrCallable,
    t: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
  """Either pass-through the float value or call a function

  Args:
    foc (float or callable): Either a float or a callable that
      accepts an array and returns an array
    t (array): Argument for function :data:`foc`.
      Required if :data:`foc` is callable
    out (array): Optional. Array to use for storing results

  Returns:
    float or array: Either :data:`foc(t)` or :data:`foc`"""
  if callable(foc):
    if t is None:
      raise ValueError(
          "_float_or_call() argument t cannot be None when foc is callable")
    try:
      return foc(t, out=out)
    except TypeError:
      return foc(t)
  return foc


class Beat:
  """Model for beating partials

  Args:
    a0 (float or callable): Amplitude of first partial.
      It can be a function of time
    a1 (float or callable): Amplitude of second partial.
      It can be a function of time
    f0 (float or callable): Frequency of first partial.
      It can be a function of time
    f1 (float or callable): Frequency of second partial.
      It can be a function of time
    p0 (float or callable): Phase of first partial.
      It can be a function of time
    p1 (float or callable): Phase of second partial.
      It can be a function of time"""

  def __init__(self,
               a0: FloatOrCallable = 1,
               a1: FloatOrCallable = 1,
               f0: FloatOrCallable = 0.95,
               f1: FloatOrCallable = 1.05,
               p0: FloatOrCallable = 0,
               p1: FloatOrCallable = 0):
    self._a0 = a0
    self._a1 = a1
    self._f0 = f0
    self._f1 = f1
    self._p0 = p0
    self._p1 = p1

    # --- Define computational graph ------------------------------------------
    self._graph = {"t": pg.Variable("t")}
    self._define_base_graph()._define_am()._define_fm()
    self._graph["x"] = np2pg.multiply(self._graph["am"],
                                      np2pg.cos(self._graph["pm"]))
    # -------------------------------------------------------------------------

  def _define_base_graph(self):
    """Define basic derived variables"""
    for k, i in itertools.product(("a", "f", "p"), range(2)):
      self._graph[f"{k}{i}"] = pg.op(
          utils.NamedObject(
              functools.partial(_float_or_call, getattr(self, f"_{k}{i}")),
              f"{k}{i}"))(self._graph["t"])
    for k in ("a", "f", "p"):
      self._graph[f"{k}_oln"] = np2pg.semisum(self._graph[f"{k}0"],
                                              self._graph[f"{k}1"])
      self._graph[f"{k}_hat"] = np2pg.semidiff(self._graph[f"{k}0"],
                                               self._graph[f"{k}1"])
    self._graph["w_hat"] = np2pg.multiply(2 * np.pi, self._graph["f_hat"])
    self._graph["w_oln"] = np2pg.multiply(2 * np.pi, self._graph["f_oln"])
    return self

  def _define_am(self):
    """Define Amplitude Modulation variables"""
    self._graph["a_oln2"] = np2pg.square(self._graph["a_oln"])
    self._graph["a_hat2"] = np2pg.square(self._graph["a_hat"])

    phase = np2pg.multiply(self._graph["w_hat"], self._graph["t"])
    phase = np2pg.add(phase, self._graph["p_hat"])
    modulant = np2pg.square(np2pg.cos(phase))
    am_range = np2pg.subtract(self._graph["a_oln2"], self._graph["a_hat2"])
    self._graph["alpha2"] = np2pg.add(np2pg.multiply(am_range, modulant),
                                      self._graph["a_hat2"])
    self._graph["am"] = np2pg.multiply(np2pg.sqrt(self._graph["alpha2"]), 2)
    return self

  def _define_fm(self):
    """Define Frequency Modulation variables"""
    # Instantaneous frequency
    f = np2pg.multiply(self._graph["a_oln"], self._graph["a_hat"])
    f = np2pg.multiply(f, self._graph["w_hat"])
    f = np2pg.true_divide(f, self._graph["alpha2"])
    f = np2pg.add(f, self._graph["w_oln"])
    self._graph["fm"] = f
    # Initial phase
    y0 = np2pg.multiply(self._graph["a_hat"], np2pg.sin(self._graph["p_hat"]))
    x0 = np2pg.multiply(self._graph["a_oln"], np2pg.cos(self._graph["p_hat"]))
    p0 = np2pg.add(self._graph["p_oln"], np2pg.arctan2(y0, x0))
    # Instantenous Phase
    p = pg.op(integrate.cumulative_trapezoid)(f, self._graph["t"], initial=0)
    p = np2pg.add(p, p0)
    self._graph["pm"] = p
    return self

  @property
  def variables(self) -> Tuple[str, ...]:
    """List of the names of the variables that can be computed by the model"""
    return tuple(self._graph)

  def compute(self, t: np.ndarray,
              output: Union[str, Iterable[str]]) -> List[np.ndarray]:
    """Compute variables

    Args:
      t (array): Time axis
      output: Names of the variables to compute"""
    if isinstance(output, str):
      return self.compute(t, output=[output])[0]
    return pg.evaluate(list(map(self._graph.__getitem__, output)),
                       {self._graph["t"]: t})

  def __getattr__(self, key: str):
    if key in self._graph:
      return functools.partial(self.compute, output=key)
    else:
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{key}'")


class ExponentialDecay:
  r"""Exponentially decaying function
  :math:`f(t) = a\cdot e^{-\frac{2}{d}t}`

  Args:
    a (float): Amplitude at time :math:`t=0`
    d (float): Decay in seconds"""

  def __init__(self, a: float, d: float):
    self._a = a
    self._k = -2 / d

  @utils.numpy_out(method=True, dtype=float)
  def __call__(self, t: np.ndarray, out: Optional[np.ndarray] = None):
    """Compute function at time :data:`t`

    Args:
      t (array): Time-steps at which to evaluate the function
      out (array): Optional. Array to use for storing results

    Returns:
      array: Function evaluated at time :data:`t`"""
    np.multiply(self._k, t, out=out)
    np.exp(out, out=out)
    np.multiply(self._a, out, out=out)
    return out


class ModalBeat(Beat):
  """Model for beating exponentially-decaying partials

  Args:
    a0 (float): Amplitude of first partial
    a1 (float): Amplitude of second partial
    f0 (float): Frequency of first partial
    f1 (float): Frequency of second partial
    d0 (float): Decay of the first partial
    d1 (float): Decay of the second partial
    p0 (float): Phase of first partial
    p1 (float): Phase of second partial"""

  def __init__(self,
               a0: float = 1,
               a1: float = 1,
               f0: float = 0.95,
               f1: float = 1.05,
               d0: float = 1,
               d1: float = 1,
               p0: float = 0,
               p1: float = 0):
    super().__init__(f0=f0,
                     f1=f1,
                     p0=p0,
                     p1=p1,
                     a0=ExponentialDecay(a=a0, d=d0),
                     a1=ExponentialDecay(a=a1, d=d1))
