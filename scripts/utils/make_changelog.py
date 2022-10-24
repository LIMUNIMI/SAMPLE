"""Utility for creating the :data:`docs/source/changelog.rst` file"""
import argparse
import functools
import itertools
import logging
import os
import sys
from typing import Tuple

from chromatictools import cli

logger = logging.getLogger("SAMPLE-make-changelog")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    default_out = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "docs", "source",
                     "changelog.rst"))
    default_in = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "changelog"))
    self.add_argument("--output",
                      "-O",
                      metavar="PATH",
                      default=default_out,
                      help="Output file path")
    self.add_argument("--input",
                      "-I",
                      metavar="PATH",
                      default=default_in,
                      help="Input directory path")
    self.add_argument("--dry-run",
                      action="store_true",
                      help="Do not output a file, only print its contents")
    self.add_argument("--separator",
                      "-s",
                      metavar="SEP",
                      default="_",
                      help="Separator for version numbers")
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP evaluation script

    Args:
      argv (tuple): CLI arguments

    Returns:
      Namespace: Parsed arguments"""
    args = self.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)
    logger.setLevel(args.log_level)
    logger.debug("Args: %s", args)
    return args


def _version_compare(u: Tuple[int, ...], v: Tuple[int, ...]) -> int:
  """Compare version numbers"""
  for i, j in itertools.zip_longest(u, v):
    if i == j:
      if i is None:
        break
      else:
        continue
    if i is None:
      return -1
    if j is None:
      return 1
    return -1 if i < j else 1
  return 0


_VersionKey = functools.cmp_to_key(_version_compare)


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  files = os.listdir(args.input)
  logger.debug("Files: %s", files)
  relpath = os.path.relpath(args.input, os.path.dirname(args.output))
  logger.debug("Relative path of inputs: %s", relpath)

  d = {}
  for f in files:
    v = "".join(itertools.dropwhile(str.isalpha,
                                    os.path.splitext(f)[0])).split(
                                        args.separator)
    try:
      v = tuple(map(int, v))
    except ValueError:
      logger.error("Error getting version number from file name: %s", f)
      continue
    s = os.path.join(relpath, f)
    d[s] = v
  d = sorted(d.items(), key=lambda t: _VersionKey(t[1]), reverse=True)
  logger.debug("Files and versions: %s", d)

  outs = ["Changelog\n=========\n"]
  outs.extend(f".. mdinclude:: {f}" for f, _ in d)
  outs.append("")
  outs = "\n".join(outs)
  if args.dry_run:
    print(outs)
  else:
    with open(args.output, mode="w", encoding="utf-8") as f:
      f.write(outs)
