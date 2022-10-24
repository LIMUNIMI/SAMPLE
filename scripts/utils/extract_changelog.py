"""Retrieve changelog for the correct version"""
import argparse
import contextlib
import itertools
import os
import sys

from chromatictools import cli

import sample


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--version",
                      metavar="TAG",
                      default=sample.__version__,
                      help="Version tag")
    self.add_argument(
        "--dir",
        metavar="PATH",
        default="changelog",
        help="Changelog directory",
    )
    self.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output file path",
    )


@contextlib.contextmanager
def open_if_any(filepath, *args, encoding: str = "utf-8", **kwargs):
  """Open file if filepath is not :data:`None`"""
  if filepath is None:
    yield None
  else:
    with open(filepath, *args, encoding=encoding, **kwargs) as f:
      yield f


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Run script"""
  args, _ = ArgParser(description=__doc__).parse_known_args(argv)
  filename = "".join(
      itertools.takewhile(lambda c: c == "_" or c.isdigit(),
                          args.version.replace(".", "_")))
  filepath = os.path.join(args.dir, f"v{filename}.md")
  with open(filepath, mode="r", encoding="utf-8") as fin:
    with open_if_any(args.output, mode="w") as fout:
      i = False
      for r in fin:
        if i:
          if fout is None:
            print(r, end="")
          else:
            fout.write(r)
        else:
          i = True
