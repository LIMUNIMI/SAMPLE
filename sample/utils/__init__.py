"""Utility functions"""
import functools
import inspect
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import paragraph as pg


def comma_join_quote(it: Iterable) -> str:
  """Join strings with a comma and surround each element with quotes

  Args:
    it (iterable): Iterable of elements to join

  Returns:
    str: Joined string"""
  return ", ".join(f"'{s}'" for s in map(str, it))


def add_keyword_arg(func: Optional[Callable] = None,
                    sig_func: Optional[Callable] = None,
                    name: Optional[str] = None,
                    **kwargs):
  """Add a keyword argument to a function signature

  Args:
    func (callable): Function to affect. If :data:`None`, then return a
      decorator for decorating a function
    sig_func (callable): Function from which to get the signature.
      If :data:`None`, then use the signature of the decorated function
    name (str): Parameter name
    kwargs: Keyword arguments for :class:`inspect.Parameter`

  Returns:
    callable: Decorated function"""
  if func is None:
    return functools.partial(add_keyword_arg,
                             sig_func=sig_func,
                             name=name,
                             **kwargs)
  if sig_func is None:
    sig_func = func
  sig = inspect.signature(sig_func)
  params = list(filter(lambda p: p.name != name, sig.parameters.values()))
  i = len(params)
  if i > 0 and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
    i -= 1
  params.insert(
      i,
      inspect.Parameter(kind=inspect.Parameter.KEYWORD_ONLY,
                        name=name,
                        **kwargs))
  func.__signature__ = sig.replace(parameters=params)
  return func


def function_with_variants(func: Optional[Callable] = None,
                           key: str = "type",
                           default: str = "default",
                           this: Optional[str] = None):
  """Declare that a function has variants

  Args:
    func (callable): Function to affect. If :data:`None`, then return a
      decorator for decorating a function
    key (str): Keyword argument name for choosing the function variant.
      Default is :data:`"type"`
    default (str): Default keyword argument value for choosing the function
      variant. Default is :data:`"default"`
    this (str): Keyword argument value for choosing the decorated function as
      the variant. If :data:`None` (default), then the decorated function is
      never

  Returns:
    callable: Decorated function"""
  if func is None:
    return functools.partial(function_with_variants,
                             key=key,
                             default=default,
                             this=this)
  func_dict = {}
  if this is not None:
    func_dict[this] = func

  @add_keyword_arg(name=key, default=default, annotation=str)
  @functools.wraps(func)
  def func_(*args, **kwargs):
    k = kwargs.pop(key, default)
    try:
      foo = func_dict[k]
    except KeyError as e:
      raise ValueError(
          f"{func.__name__}: unsupported option '{k}' for argument '{key}'. "
          f"Supported options are: {comma_join_quote(func_dict)}") from e
    return foo(*args, **kwargs)

  func_._variants_dict = func_dict  # pylint: disable=W0212
  return func_


def function_variant(main_func: Callable,
                     key: str,
                     this: Optional[Callable] = None):
  """Register this function as a variant of a function with
  variants (see :func:`function_with_variants`)

  Args:
    main_func (callable): Function with variants
    key (str): Keyword argument value for choosing this function variant.
    this (callable): Function to register. If :data:`None`, then return a
      decorator for decorating a function

  Returns:
    callable: Decorated function"""
  if this is None:
    return functools.partial(function_variant, main_func, key)
  main_func._variants_dict[key] = this  # pylint: disable=W0212
  return this


def _result_type(*args):
  """Custom wrapper for :func:`numpy.result_type`"""
  ts = []
  for a in args:
    try:
      t = [np.result_type(a)]
    except TypeError:
      t = map(np.result_type, a)
    ts.extend(t)
  return np.result_type(*ts)


def numpy_id(a: np.ndarray) -> int:
  """Get the ID of the memory location of a numpy array

  Args:
    a (ndarray): Numpy array to inspect

  Returns:
    int: The memory location ID"""
  return a.__array_interface__["data"][0]


def numpy_out(func: Optional[Callable] = None,
              method: bool = False,
              key: str = "out",
              dtype_key: str = "dtype",
              dtype: Optional[np.dtype] = None,
              dtype_promote: bool = True):
  """Automatically handle the preprocesing of the :data:`out` argument for
  a numpy-like function

  Args:
    func (callable): Numpy-like function
    method (bool): Set this to :data:`True` when wrapping a method
    key (str): Argument name
    dtype_key (str): Argument name for dtype. Default is :data:`"dtype"`
    dtype (dtype): Default dtype of output
    dtype_promote (bool): If :data:`True` (default), then promote
      the output dtype, otherwise enforce default, unless specified by the
      user of the decorated function

  Returns:
    callable: Decorated function"""
  if func is None:
    return functools.partial(numpy_out,
                             method=method,
                             key=key,
                             dtype_key=dtype_key,
                             dtype=dtype,
                             dtype_promote=dtype_promote)
  if dtype is None:

    def _get_dtype_inner(args):
      return _result_type(*args)
  elif dtype_promote:

    def _get_dtype_inner(args):
      return _result_type(dtype, *args)
  else:

    def _get_dtype_inner(_):
      return dtype

  def _get_dtype(args, kwargs):
    return kwargs.pop(dtype_key) if dtype_key in kwargs else _get_dtype_inner(
        args)

  if method:

    def _get_args(args):
      return args[:1], args[1], args[2:]
  else:

    def _get_args(args):
      return (), args[0], args[1:]

  @functools.wraps(func)
  def func_(*args, **kwargs):
    nd = None
    s, a, args = _get_args(args)
    if kwargs.get(key, None) is None:
      nd = np.ndim(a)
      kwargs[key] = np.empty(() if nd == 0 else np.shape(a),
                             dtype=_get_dtype((a, *args), kwargs))
    out = func(*s, a, *args, **kwargs)
    return out if nd is None or nd != 0 else out[()]

  return func_


class NamedObjectMeta(type):
  """Metaclass for named objects"""

  def __call__(cls, obj, *args, **kwargs) -> type:
    """Initialize named object"""
    f = NamedCallable if callable(obj) and cls is NamedObject else cls
    self = f.__new__(f, obj, *args, **kwargs)
    self.__init__(obj, *args, **kwargs)
    return self


class NamedObject(metaclass=NamedObjectMeta):
  """Wrap an object and give it a name

  Args:
    obj (Any): Object to wrap
    name (str): Name for object"""

  def __init__(self, obj: Any, name: Optional[str] = None):
    self.obj = obj
    if name is not None:
      self.__name__ = name

  # def __getattr__(self, key: str):
  #   """Get attribute from inner object"""
  #   return getattr(self.obj, key)


class NamedCallable(NamedObject):
  """Wrap a callable and give it a name"""

  def __call__(self, *args, **kwargs):
    """Call inner callable"""
    return self.obj(*args, **kwargs)


class Numpy2Paragraph:
  """Class for converting numpy functions to paragraph operators"""

  def __init__(self):
    self._d = {}

  @staticmethod
  @numpy_out(dtype=float)
  def _semisum(x: np.ndarray,
               y: np.ndarray,
               out: Optional[np.ndarray] = None,
               **kwargs):
    """Half of the sum

    Args:
      x (array): First input
      y (array): Second input
      out (array): Optional. Array to use for storing results

    Returns:
      array: Half of the sum of the inputs"""
    out = np.add(x, y, out=out, **kwargs)
    return np.true_divide(out, 2, out=out, **kwargs)

  @staticmethod
  @numpy_out(dtype=float)
  def _semidiff(x: np.ndarray,
                y: np.ndarray,
                out: Optional[np.ndarray] = None,
                **kwargs):
    """Half of the difference

    Args:
      x (array): First input
      y (array): Second input
      out (array): Optional. Array to use for storing results

    Returns:
      array: Half of the difference of the inputs"""
    out = np.subtract(x, y, out=out, **kwargs)
    return np.true_divide(out, 2, out=out, **kwargs)

  semisum = pg.op(NamedObject(_semisum.__get__(object), name="semisum"))  # pylint: disable=E1120
  semidiff = pg.op(NamedObject(_semidiff.__get__(object), name="semidiff"))  # pylint: disable=E1120

  def __getattr__(self, key: str) -> pg.op:
    """Make an operator out of a numpy function"""
    if key not in self._d:
      self._d[key] = pg.op(getattr(np, key))
    return self._d[key]


np2pg = Numpy2Paragraph()
