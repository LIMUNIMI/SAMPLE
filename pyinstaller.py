"""Pyinstaller run file"""
import PyInstaller.__main__
import itertools
from chromatictools import cli
import os
import librosa
from typing import Tuple, Iterable
from types import ModuleType


def module_dir(mod: ModuleType) -> str:
  """Get the install directory of a module

  Args:
    mod: Module

  Returns:
    str: Install path"""
  return os.path.dirname(mod.__file__)


def module_data_path(mod: ModuleType, *path: Tuple[str]) -> str:
  """Get a :data:`--add-data` path specification for module files

  Args:
    mod: Module
    path: Path nodes (strings)

  Return:
    str: Path specification for :data:`--add-data` option"""
  mdir = module_dir(mod)
  name = os.path.basename(mdir)
  d_from = os.path.join(mdir, *path)
  d_to = os.path.join(name, *path)
  return os.pathsep.join((d_from, d_to))


def hidden_imports(it: Iterable[str]) -> Tuple[str, ...]:
  """Get cli arguments for hidden imports

  Args:
    it: Iterable of module names

  Returns:
    tuple of str: Cli arguments for hidden imports"""
  return tuple(
    itertools.chain.from_iterable(
      zip(
        itertools.repeat("--hidden-import"),
        it
      )
    )
  )


def module_data(
  it: Iterable["Tuple[ModuleType, str, ...]"]
) -> Tuple[str, ...]:
  """Get cli arguments for hidden imports

  Args:
    it: Iterable of tuples. Every tuple starts with a module
      and continues with path nodes of the data files

  Returns:
    tuple of str: Cli arguments for adding data"""
  return tuple(
    itertools.chain.from_iterable(
      zip(
        itertools.repeat("--add-data"),
        map(lambda t: module_data_path(*t), it)
      )
    )
  )


@cli.main(__name__)
def main():
  """Run pyinstaller"""
  PyInstaller.__main__.run([
    "sample-gui.py", "-F",
    "-n", "SAMPLE",
    *hidden_imports((
      "sklearn.utils._weight_vector",
      "PIL._tkinter_finder",
    )),
    *module_data((
      (librosa, "util", "example_data"),
    )),
  ])
