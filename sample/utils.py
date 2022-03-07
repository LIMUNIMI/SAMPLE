"""Utility functions"""
import functools
import inspect
from typing import Callable, Iterable, Optional

import numpy as np


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


def numpy_out(func: Optional[Callable] = None, key: str = "out"):
  """Automatically handle the preprocesing of the :data:`out` argument for
  a numpy-like function

  Args:
    func (callable): Numpy-like function
    key (str): Argument name

  Returns:
    callable: Decorated function"""
  if func is None:
    return functools.partial(numpy_out, key=key)
  @functools.wraps(func)
  def func_(a, *args, **kwargs):
    nd = None
    if key not in kwargs:
      nd = np.ndim(a)
      kwargs[key] = np.empty(() if nd == 0 else np.shape(a))
    out = func(a, *args, **kwargs)
    return out if nd is None or nd != 0 else out[()]
  return func_
