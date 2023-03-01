"""Pyinstaller run file"""
import contextlib
import io
import os

import PIL
import PyInstaller.__main__
from chromatictools import cli

from sample import vid


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
    cmd = [
        "sample_gui.py",
        "-F",
        f"--icon={icon_fpath}",
        "-n",
        "SAMPLE",
    ]
    print(*cmd)
    PyInstaller.__main__.run(cmd)
