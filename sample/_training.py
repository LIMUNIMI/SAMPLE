"""Make SAMPLE trainable in multiprocessing"""
import copy
import contextlib
import functools
import itertools
import multiprocessing
import multiprocessing.managers
from typing import Generator, Callable, Iterable, Optional

import sample.utils
import sample.utils.learn

mp = multiprocessing
utils = sample.utils


class _OptionalStorageManager(mp.managers.SyncManager):
  """Multiprocessing manager for :class:`sample.utils.learn.OptionalStorage`"""


_OptionalStorageManager.register(
    "OptionalStorage",
    utils.learn.OptionalStorage,
    exposed=(
        *mp.managers.public_methods(utils.learn.OptionalStorage),
        "__getitem__",
    ))


class FitArgs:
  """Arguments for :func:`SAMPLE.fit`"""

  def __init__(self, starmap=itertools.starmap, progressbar=None) -> None:
    self.starmap = starmap
    self.progressbar = progressbar

  def starmap_progress(self,
                       func: Callable,
                       it: Iterable[Iterable],
                       tot: Optional[int] = None):
    """Apply starmap and update progress"""
    if tot is not None:
      self.progress_start(tot)
    for i in self.starmap(func, it):
      yield i
      self.progress_update()
    if tot is not None:
      self.progress_stop()

  def progress_start(self, maximum: int, value: int = 0):
    """Progress start method stub"""

  def progress_update(self, value: Optional[float] = None):
    """Progress update method stub"""

  def progress_stop(self):
    """Progress stop method stub"""


def _progress_starmap(pool: mp.Pool,
                      fit_args: FitArgs,
                      func: Callable,
                      iterable: Iterable[Iterable],
                      refresh_time: float = 0.01,
                      **kwargs):
  """Utility function for using a process pool asynchronously and updating
  progress in the main thread. Avoids to hit some limitations imposed by tk"""
  results = [pool.apply_async(func, args=args, **kwargs) for args in iterable]
  not_ready = results.copy()
  while not_ready:
    fit_args.progress_update(len(results) - len(not_ready))
    if refresh_time:
      not_ready[0].wait(refresh_time)
    not_ready = [r for r in not_ready if not r.ready()]
  return (r.get() for r in results)


@contextlib.contextmanager
def sample_training_context(model: "sample.sample.SAMPLE",
                            n_jobs: Optional[int] = 0,
                            mp_manager: Optional[
                                mp.managers.BaseManager] = None,
                            pool: Optional[mp.Pool] = None,
                            **kwargs) -> Generator[FitArgs, None, None]:
  """Context manager for training a SAMPLE model in parallel transparently.

  Args:
    model (SAMPLE): SAMPLE model to train
    n_jobs (int): Number of parallel processes (ignored if a
      :data:`pool` is specified)
    mp_manager (multiprocessing.managers.BaseManager): Multiprocessing manager
      for optional storage
    pool (multiprocessing.Pool): Process pool

  Yields:
    SAMPLE fit arguments"""
  # Avoid _fit_args appearing in signatures, since it's considered
  # an implementation detail
  fit_args = kwargs.pop("_fit_args", FitArgs())
  if pool is None and n_jobs == 0:
    # No multiprocessing
    yield fit_args
    return
  if mp_manager is None:
    with _OptionalStorageManager() as m:
      with sample_training_context(model=model,
                                   mp_manager=m,
                                   pool=pool,
                                   n_jobs=n_jobs,
                                   _fit_args=fit_args) as fit_args_:
        yield fit_args_
        return
  if pool is None:
    with mp.Pool(processes=n_jobs) as p:
      with sample_training_context(model=model,
                                   mp_manager=mp_manager,
                                   pool=p,
                                   n_jobs=n_jobs,
                                   _fit_args=fit_args) as fit_args_:
        yield fit_args_
        return
  with _wrap_as_managed(mp_manager, model, "beat_decisor__intermediate",
                        "sinusoidal__intermediate"):
    fit_args_ = copy.copy(fit_args)
    if pool is not None:
      fit_args_.starmap = functools.partial(_progress_starmap, pool, fit_args_)
    yield fit_args_


@contextlib.contextmanager
def _wrap_as_managed(mp_manager: mp.managers.BaseManager,
                     model: "sample.sample.SAMPLE", key, *more_keys):
  """Temporarily patch a member object as a managed instance"""
  if more_keys:
    with _wrap_as_managed(mp_manager, model, key):
      with _wrap_as_managed(mp_manager, model, *more_keys):
        yield
    return
  try:
    original = model.get_params(deep=True)[key]
  except KeyError:
    yield
  else:
    managed = getattr(mp_manager,
                      type(original).__name__)(**original.get_params())
    try:
      model.set_params(**{key: managed})
      yield
      if isinstance(original, utils.learn.OptionalStorage):
        original.cache_ = managed.get_state()
    finally:
      model.set_params(**{key: original})
