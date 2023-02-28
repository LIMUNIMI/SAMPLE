"""Machine learning utilities"""
import copy
import functools
import itertools
from typing import Any, Callable, Dict, Optional, TypeVar, overload

from sklearn import base

T = TypeVar("T")
DefaultFunction = Callable[[Any], Any]


class _DefaultProperty(property):
  """Property that falls back to building a default value on set

  Args:
    name (str): Property name
    default_fn (callable): Function for building a default value.
      It should take the parent object as a positional argument.
    fdef (callable): Defaulter function. See :meth:`defaulter`
    **kwargs: Keyword arguments for :class:`property`"""

  _FORBIDDEN_NAMES: Dict[Optional[str], str] = {
      None: "",
      "<lambda>": " because of possible conflicts with other lambda functions",
      "_": " because it is conventionally used as a placeholder",
  }

  class DocumentedCallable:
    """Wrapper for a callable and its docstring. This class avoids overwriting
    the callable's :data:`__doc__` attribute

    Args:
      func (callable): Callable
      doc (str): Docstring"""

    def __init__(self, func: Callable, doc: Optional[str] = None):
      self.func = func
      self.doc = doc

    def __call__(self, *args: Any, **kwargs: Any):
      return self.func(*args, **kwargs)

    @property
    def __doc__(self) -> Optional[str]:
      if self.doc is None:
        return getattr(self.func, "__doc__", None)
      return self.doc

  def __init__(self,
               default_fn: DefaultFunction,
               name: Optional[str] = None,
               **kwargs) -> None:
    if name is None:
      name = getattr(default_fn, "__name__", None)
    if name in self._FORBIDDEN_NAMES:
      why = self._FORBIDDEN_NAMES[name]
      raise ValueError(f"default_property(): name '{name}' is forbidden{why}"
                       ". Please, specify a different 'name'")
    self._attr_name = f"_default_property_{name}"
    self._default_fn = default_fn

    if "doc" not in kwargs:
      kwargs["doc"] = getattr(default_fn, "__doc__", None)
    for k in ("fget", "fset", "fdef"):
      if k not in kwargs:
        kwargs[k] = getattr(self, f"_default_{k}")
    self._doc = kwargs.pop("doc")
    # This wrapper allows sphinx to get the correct docstring
    kwargs["fget"] = self.DocumentedCallable(kwargs["fget"], self._doc)
    self.defaulter(kwargs.pop("fdef"))
    super().__init__(**kwargs)

  def defaulter(self, fdef: Optional[Callable[[Any, Any], bool]] = None):
    """Assign a default-checker function to the property

    Args:
      fdef (callable): Function that takes the parent object and
        the value to-be-set and outputs :data:`True` if the default
        value should be set, instead. If :data:`None`, then use a
        constant function that always returns :data:`False`"""
    self.fdef = self._constant_fdef if fdef is None else fdef

  def _default_fget(self, parent):
    """Default fget function"""
    return getattr(parent, self._attr_name)

  def _default_fset(self, parent, v):
    """Default fset function"""
    if self.fdef(parent, v):
      v = self._default_fn(parent)
    return setattr(parent, self._attr_name, v)

  @staticmethod
  def _constant_fdef(parent, v) -> bool:  # pylint: disable=W0613
    """Constant fdef function"""
    return False

  @staticmethod
  def _default_fdef(parent, v) -> bool:  # pylint: disable=W0613
    """Default fdef function"""
    return v is None


@overload
def default_property(**kwargs) -> Callable[[DefaultFunction], _DefaultProperty]:
  ...  # pragma: no cover


@overload
def default_property(default_fn: DefaultFunction, **kwargs) -> _DefaultProperty:
  ...  # pragma: no cover


def default_property(default_fn: Optional[DefaultFunction] = None, **kwargs):
  # pylint: disable=C0301 (line-too-long unavoidable because of doctest)
  """Default property attribute. Can be used as a decorator.
  It's meant to be used to avoid unexpected sharing of instances between
  different objects when using default argument values.
  Unexpected sharing of instances can be a problem when those instances
  not only encapsulate initialization arguments but also store some state,
  such as the parameters learned during training.

  Example:
    >>> # Let's define a class for member objects
    >>> class Member:
    ...   def __init__(self, field=0):
    ...     self.field = field
    >>> # And a class that correctly uses the default_property decorator
    >>> from sample.utils.learn import default_property
    >>> class Good:
    ...   def __init__(self, member=None, field=0):
    ...     self.member = member
    ...     self.field = field
    ...   @default_property
    ...   def member(self):
    ...     return Member()
    >>> # Instances created with default arguments do not share
    >>> # the same instance of the member object
    >>> Good().member is not None
    True
    >>> Good().member is not Good().member
    True
    >>> # Unless we explicitly provide a value
    >>> x = Member()
    >>> Good(x).member is Good(x).member
    True
    >>> # A class with default values will share
    >>> # the same instance of the member object
    >>> class Bad:
    ...   def __init__(self, member=Member(), field=0):
    ...     self.member = member
    ...     self.field = field
    >>> Bad().member is Bad().member
    True
    >>> # Note that the member object is created on set, and not on get
    >>> g = Good()
    >>> g.member is g.member
    True
    >>> # You can also change the condition that triggers the default
    >>> # (by default, it is triggered when the input is None)
    >>> # E.g.: this function sets the default when the input value is not an
    >>> # instance of the Member class
    >>> Good.member.defaulter(lambda _, v: not isinstance(v, Member))
    >>> isinstance(Good(g).member, Member)
    True
    >>> Good(g).member is not g
    True
    >>> # While allowing assignments to get trough for valid inputs
    >>> Good(x).member is x
    True
    >>> # Default values can be deactivated using the constant defaulter
    >>> Good.member.defaulter()
    >>> Good().member is None
    True
    >>> # Default properties can be defined dynamically
    >>> # Some callables such as lambdas and partials will
    >>> # require an additional 'name' argument
    >>> try:
    ...   Good.member = default_property(lambda _: {})
    ... except ValueError as e:
    ...   print(e)
    default_property(): name '<lambda>' is forbidden because of possible conflicts with other lambda functions. Please, specify a different 'name'
    >>> Good.member = default_property(name="member")(lambda _: {})
    >>> isinstance(Good().member, dict) and not Good().member
    True
    >>> Good().member is not Good().member
    True"""
  if default_fn is None:
    return functools.partial(_DefaultProperty, **kwargs)
  return _DefaultProperty(default_fn=default_fn, **kwargs)


class OptionalStorage(base.BaseEstimator):
  """Storage that can be deactivated. It's main use is in optionally saving
  intermediate values in machine learning processes

  Args:
    save (bool): Determines if the storage is actually operational

  Example:
    >>> # Let's define our own regressor
    >>> from sample.utils.learn import *
    >>> from sklearn.base import BaseEstimator
    >>> from sklearn.utils import check_X_y
    >>> import numpy as np
    >>> class CoolRegression(base.BaseEstimator):
    ...   def __init__(self, storage=None, **kwargs):
    ...     # This object will memorize some intermediate values
    ...     self.storage = storage
    ...     self.set_params(**kwargs)
    ...   @default_property
    ...   def storage(self):
    ...     return OptionalStorage()
    ...   def fit(self, X: np.ndarray, y: np.ndarray):
    ...     # Delete state from eventual previous runs
    ...     self.storage.reset()
    ...     check_X_y(X, y)
    ...     xcorr = X.T @ X
    ...     # Save the X.T @ X square matrix
    ...     self.storage.append("xcorr", xcorr)
    ...     pinv = np.linalg.pinv(xcorr)
    ...     # Save the pseudo-inverse matrix
    ...     self.storage.append("pinv", pinv)
    ...     self.coeffs_ = pinv @ X.T @ y
    ...     return self
    >>> # Let's simulate some data points for testing
    >>> n = 32
    >>> x = np.random.randn(n, 2) * 24
    >>> y = x[:, 0] * 4 - x[:, 1] * 0.5 + np.random.randn(n) * 0.5
    >>> # By default, the storage is inactive
    >>> cr = CoolRegression().fit(x, y)
    >>> try:
    ...   cr.storage.cache_.keys()
    ... except AttributeError as e:
    ...   print(e)
    'OptionalStorage' object has no attribute 'cache_'
    >>> # If save==True, then the storage is active
    >>> cr = CoolRegression(storage__save=True).fit(x, y)
    >>> list(cr.storage.cache_.keys())
    ['xcorr', 'pinv']
    >>> # And the matrices have the expected shape
    >>> cr.storage["xcorr"][0].shape
    (2, 2)
    >>> cr.storage["pinv"][0].shape
    (2, 2)"""

  def __init__(self, save: bool = False):
    self.save = save

  def __getitem__(self, key):
    return getattr(self, "cache_", {})[key]

  def get_state(self, deepcopy: bool = True):
    """Retrieve the storage state

    Args:
      deepcopy (bool): If :data:`True`, make a deep copy if the state

    Returns:
      dict: The state"""
    s = self.cache_ if hasattr(self, "cache_") else {}
    return copy.deepcopy(s) if deepcopy else s

  def reset(self):
    """Reset memory"""
    if hasattr(self, "cache_"):
      del self.cache_

  def append(self, key: str, value: T, index: Optional[int] = None) -> T:
    """Append variable in cache if :data:`self.save` is True

    Arguments:
      key (str): Data name
      value: Data
      index (int): Optional. The value will be added to a list,
        at the specified index

    Returns:
      object: The input value"""
    if self.save:
      if not hasattr(self, "cache_"):
        self.cache_ = {}
      if key not in self.cache_:
        self.cache_[key] = []
      if index is None:
        self.cache_[key].append(value)
      else:
        if index >= len(self.cache_[key]):
          self.cache_[key].extend(
              itertools.repeat(None, index + 1 - len(self.cache_[key])))
        self.cache_[key][index] = value
    return value

  __call__ = append
