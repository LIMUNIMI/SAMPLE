"""Pyinstaller run file"""
import contextlib
import io
import itertools
import os
from types import ModuleType
from typing import Iterable, Tuple

import librosa
import PIL
import PyInstaller.__main__
from chromatictools import cli

from sample import vid


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
      itertools.chain.from_iterable(zip(itertools.repeat("--hidden-import"),
                                        it)))


def module_data(it: Iterable["Tuple[ModuleType, str, ...]"]) -> Tuple[str, ...]:
  """Get cli arguments for hidden imports

  Args:
    it: Iterable of tuples. Every tuple starts with a module
      and continues with path nodes of the data files

  Returns:
    tuple of str: Cli arguments for adding data"""
  return tuple(
      itertools.chain.from_iterable(
          zip(itertools.repeat("--add-data"),
              map(lambda t: module_data_path(*t), it))))


@contextlib.contextmanager
def export_icon(fname: str):
  """Context manager for exporting logo icon. The icon is removed on exit

  Args:
    fname: Icon file path

  Yields:
    str: Icon file path"""
  with io.BytesIO() as buf:
    vid.icon_plt_fn(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    img.save(fname)
  yield fname
  os.remove(fname)


@cli.main(__name__)
def main():
  """Run pyinstaller"""
  with export_icon("SAMPLE.ico") as icon_fpath:
    PyInstaller.__main__.run([
        "sample_gui.py",
        "-F",
        f"--icon={icon_fpath}",
        "-n",
        "SAMPLE",
        *hidden_imports((
            "sklearn.neighbors._partition_nodes",
            "sklearn.utils._heap",
            "sklearn.utils._sorting",
            "sklearn.utils._typedefs",
            "sklearn.utils._vector_sentinel",
            "sklearn.utils._weight_vector",
            "PIL._tkinter_finder",
        )),
        *module_data(((librosa, "util", "example_data"),)),
    ])
