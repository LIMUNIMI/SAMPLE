"""Retrieve changelog for the correct version"""
from chromatictools import cli
import contextlib
import itertools
import argparse
import sample
import sys
import os


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
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--version",
                      metavar="TAG",
                      default=sample.__version__,
                      help="Version tag")
  parser.add_argument(
      "--dir",
      metavar="PATH",
      default="changelog",
      help="Changelog directory",
  )
  parser.add_argument(
      "--output",
      metavar="PATH",
      default=None,
      help="Output file path",
  )
  args, _ = parser.parse_known_args(argv)
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
