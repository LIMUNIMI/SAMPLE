"""Utility functions"""
import functools
import inspect
import warnings
from typing import (Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar,
                    Union, overload)

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


class SAMPLEDeprecationWarning(DeprecationWarning):
  """:class:`DeprecationWarning` for the SAMPLE package"""
  pass


def deprecated_argument(old_key: str,
                        new_key: Optional[str] = None,
                        convert: Optional[Callable[[Any], Tuple[str,
                                                                Any]]] = None,
                        msg: Optional[str] = None,
                        prefix: bool = False,
                        warn: bool = True):
  # pylint: disable=C0301 (line-too-long unavoidable because of doctest)
  """Wrap a function to deprecate an argument.

  Args:
    old_key (str): Deprecated argument key
    new_key (str): New argument key. Specify this if the
      only action to take is to change the argument name
    convert (callable): Function for converting the provided argument to a new
      argument key and value. Specify this for more complex scenarios
    msg (str): Extra indications to append to the deprecation warning
    prefix (bool): If :data:`True`, then interpret :data:`old_key` and
      :data:`new_key`, not as full names, but as prefixes to be matched.
      This option can only be used when :data:`new_key` is specified
    warn (bool): If :data:`True` (default), then issue a deprecation warning
      when the deprecated argument is used

  Example:
    >>> from sample.utils import deprecated_argument, warnings_simplefilter
    >>> # First of all, let's use a filter to catch and print warnings
    >>> warns = []
    >>> filt = warnings_simplefilter(action="always",
    ...                              category=DeprecationWarning,
    ...                              catch_kws=dict(record=True),
    ...                              traces=warns,
    ...                              toggle=("ignore_warnings", True))
    >>> # Let's say that this function used to have a argument 'n',
    >>> # which was too generic and we decided to change the name to 'size'
    >>> @filt
    ... @deprecated_argument("n", "size")
    ... def func(x: str, size: int = 1):
    ...   return x * size
    >>> # Positional arguments do not trigger warnings
    >>> func("x", 5)
    'xxxxx'
    >>> warns[-1]
    []
    >>> # The 'size' argument does not trigger warnings
    >>> func("x", size=5)
    'xxxxx'
    >>> warns[-1]
    []
    >>> # The 'n' argument triggers a warning
    >>> func("x", n=5)
    'xxxxx'
    >>> len(warns[-1])
    1
    >>> print(warns[-1][-1].message)
    The 'n' argument is deprecated, use 'size' instead.
    >>> # Let's say that now we want to use an argument that specifies
    >>> # the total length of the output string
    >>> # A message explains the difference in behaviour
    >>> @filt
    ... @deprecated_argument("n", "size")
    ... @deprecated_argument(
    ...     "size",
    ...     convert=lambda x, size: ("length", len(x) * size),
    ...     msg="The new 'length' argument specifies the desired total length "
    ...     "of the string and NOT the number of repetitions")
    ... def func(x: str, length: int = None):
    ...   if length is None:
    ...     return x
    ...   return (x * (length // len(x) + 1))[:length]
    >>> # This example gives the same output as with the old function
    >>> func("x", length=5)
    'xxxxx'
    >>> warns[-1]
    []
    >>> # But this shows the new behaviour
    >>> func("_-", length=5)
    '_-_-_'
    >>> warns[-1]
    []
    >>> # The keyword 'size' issues a warning but preserves the old behaviour
    >>> func("_-", size=5)
    '_-_-_-_-_-'
    >>> len(warns[-1])
    1
    >>> print(warns[-1][-1].message)
    The 'size' argument is deprecated, use 'length' instead. The new 'length' argument specifies the desired total length of the string and NOT the number of repetitions
    >>> # To deprecate an argument, we have to specify either a new key
    >>> # or a conversion function
    >>> try:
    ...   @deprecated_argument("n")
    ...   def bad(x, m=1):
    ...     return x * m
    ... except ValueError as e:
    ...   print(e)
    At least one between 'new_key' and 'convert' should not be None
    >>> # We can deprecate all arguments starting with the same prefix
    >>> @filt
    ... @deprecated_argument("array_", "", prefix=True)
    ... def full(value, size: int = 1):
    ...   return [value] * size
    >>> full(3, 5)
    [3, 3, 3, 3, 3]
    >>> warns[-1]
    []
    >>> # The deprecated argument 'array_size' will be mapped to 'size'
    >>> # and 'array_value' to 'value'
    >>> full(array_value=3, array_size=5)
    [3, 3, 3, 3, 3]
    >>> len(warns[-1])
    2
    >>> print(warns[-1][0].message)
    The 'array_value' argument is deprecated, use 'value' instead.
    >>> print(warns[-1][1].message)
    The 'array_size' argument is deprecated, use 'size' instead.
    >>> # Using a prefix requires the 'new_key' argument
    >>> try:
    ...   @deprecated_argument("array_", prefix=True)
    ...   def bad_full(value, size: int = 1):
    ...     return [value] * size
    ... except ValueError as e:
    ...   print(e)
    prefix=True should be used in conjuction with 'new_key'"""
  if new_key is None:
    if prefix:
      raise ValueError(
          "prefix=True should be used in conjuction with 'new_key'")
    if convert is None:
      raise ValueError(
          "At least one between 'new_key' and 'convert' should not be None")
  # Turn on deprecation warnings
  if warn:
    warnings.simplefilter("always", SAMPLEDeprecationWarning)

  def deprecated_argument_(func: Callable):

    @functools.wraps(func)
    def func_(*args, **kwargs):
      if prefix:
        key_map_ = {
            k: f"{new_key}{k[len(old_key):]}"
            for k in kwargs
            if k.startswith(old_key)
        }
      else:
        key_map_ = {old_key: new_key}

      for old_key_, new_key_ in key_map_.items():
        if old_key_ in kwargs:
          if warn:
            msg_ = [f"The '{old_key_}' argument is deprecated", "."]
          new_key__ = new_key_
          if convert is not None:
            new_key__, x = convert(*args, **kwargs)
            kwargs.pop(old_key_)
            kwargs[new_key__] = x
          elif new_key__ is not None:
            kwargs[new_key__] = kwargs.pop(old_key_)

            if warn:
              msg_.insert(1, f", use '{new_key__}' instead")
          if warn:
            warnings.warn(
                f"The '{old_key_}' argument is deprecated, "
                f"use '{new_key__}' instead."
                f"{'' if msg is None else ' '}{'' if msg is None else msg}",
                SAMPLEDeprecationWarning, 2)
      return func(*args, **kwargs)

    return func_

  return deprecated_argument_


F = TypeVar("F", bound=Callable)


@overload
def warnings_simplefilter(catch_kws: Optional[Dict[str, Any]] = None,
                          traces: Optional[list] = None,
                          toggle: Optional[Tuple[str, bool]] = None,
                          **kwargs) -> Callable[[F], F]:
  ...  # pragma: no cover


@overload
def warnings_simplefilter(func: F, **kwargs) -> F:
  ...  # pragma: no cover


def warnings_simplefilter(func: Optional[F] = None,
                          catch_kws: Optional[Dict[str, Any]] = None,
                          traces: Optional[list] = None,
                          toggle: Optional[Tuple[str, bool]] = None,
                          **kwargs) -> Union[F, Callable[[F], F]]:
  """Decorator for filtering warnings. For usage examples,
  see :func:`deprecated_argument`

  Args:
    func (callable): Function to decorate. If unspecified, a partial
      :func:`warnings_simplefilter` will be returned
    catch_kws (dict): Keyword arguments for :func:`warnings.catch_warnings`
    traces (list): If specified, lists yielded by
      :func:`warnings.catch_warnings` will be appended to this list
    toggle (str, bool): Key and default value for a function argument that
      controls warning filtering (when :data:`True`, warnings are filtered)
    **kwargs (dict): Keyword arguments for :func:`warnings.simplefilter`"""
  if func is None:
    for k in ("catch_kws", "traces", "toggle"):
      v = locals()[k]
      if v is not None:
        kwargs[k] = v
    return functools.partial(warnings_simplefilter, **kwargs)
  ctx_fn = functools.partial(warnings.catch_warnings,
                             **({} if catch_kws is None else catch_kws))
  flt_fn = functools.partial(warnings.simplefilter, **kwargs)

  @functools.wraps(func)
  def func_(*ars, **kws):
    if toggle is not None and not kws.pop(*toggle):
      return func(*ars, **kws)
    with ctx_fn() as y:
      flt_fn()
      if traces is not None:
        traces.append(y)
      return func(*ars, **kws)

  return func_


def default_kws(d: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
  """Helper function for setting default keyword arguments in a dict

  Args:
    d (dict): Non-default arguments
    **kwargs: Default arguments

  Returns:
    dict: A copy of the input arguments, with default arguments set

  Example:
    >>> from sample.utils import default_kws
    >>> # Let's define a function with two arguments
    >>> def foo(a, b=1):
    ...   return f"A: {a} - B: {b}"
    >>> # bar will forward foo_kws to foo, using a=1 as default value for a
    >>> def bar(start="[", end="]", foo_kws=None):
    ...   return f"{start}{foo(**default_kws(foo_kws, a=1))}{end}"
    >>> bar()
    '[A: 1 - B: 1]'
    >>> # Default values may be overwritten in the function call
    >>> bar(foo_kws={'a': 0})
    '[A: 0 - B: 1]'
    >>> # And/or non-default values, too
    >>> bar(foo_kws={'a': 0, 'b': 0})
    '[A: 0 - B: 0]'
  """
  d = {} if d is None else d.copy()
  for k, v in kwargs.items():
    d[k] = d.get(k, v)
  return d
