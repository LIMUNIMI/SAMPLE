"""Use YAPF to format Jupyter notebooks"""
import argparse
import difflib
import itertools
import logging
import os
import re
import sys
from typing import Optional, Tuple

import nbformat
from chromatictools import cli
from yapf.yapflib import yapf_api

logger = logging.getLogger("YAPF-Notebook")

_default_style_file = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".style.yapf"))
_command_pattern = re.compile(r"\s*(!|%).*")
_command_escape_comment = "  # escaped-for-yapf"
_command_escape_pattern = re.compile(r"\s*pass  # (!|%).*" +
                                     _command_escape_comment)


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("file",
                      metavar="PATH",
                      nargs="*",
                      help="Path of the notebook file(s) to format")
    self.add_argument("--style",
                      metavar="PATH",
                      **({
                          "default": _default_style_file
                      } if os.path.exists(_default_style_file) else {}),
                      help="Style file or name")
    self.add_argument("--dry-run",
                      action="store_true",
                      help="Print out diff without writing over file")
    self.add_argument("--tag-errors",
                      action="store_true",
                      help="Add comments in cells with syntax errors")
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the script

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


def _wrap_command_line(s: str) -> str:
  """Comment-out a command-line line of code in a notebook.
  Also affects magic comments

  Args:
    s (str): Line to comment

  Returns:
    str: Commented line"""
  if not _command_pattern.fullmatch(s):
    return s
  indent = "".join(itertools.takewhile(lambda c: c.isspace(), s))
  cmd = s[len(indent):]
  return f"{indent}pass  # {cmd}{_command_escape_comment}"


def _unwrap_command_line(s: str) -> str:
  """Un-comment a commented command-line line of code in a notebook.
  Also affects magic comments

  Args:
    s (str): Commented line

  Returns:
    str: Line to comment"""
  if not _command_escape_pattern.fullmatch(s):
    return s
  indent = "".join(itertools.takewhile(lambda c: c.isspace(), s))
  cmd = s[(len(indent) + 8):-len(_command_escape_comment)]
  return indent + cmd


def format_nb(filepath: str,
              dry_run: bool = False,
              tag_errors: bool = False,
              style: Optional[str] = None):
  """Format notebook

  Args:
    filepath (str): Path to the notebook file
    dry_run (bool): Print-out differences without overwriting the notebook
    tag_errors (bool): Append comments about eventual yapf errors
      into the code cells
    style (str): Style file path or name

  Returns:
    bool: Return wether or not the file has been modified"""
  logger.info("  %s", filepath)
  dots = "." * len(filepath)

  with open(filepath, mode="r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
  nbformat.validate(notebook)
  diff = []

  cells = enumerate(notebook.cells)
  cells = filter(lambda c: c[1]["cell_type"] == "code", cells)
  cells = filter(lambda c: not c[1]["source"].isspace(), cells)
  cells = filter(
      lambda c: "# yapf: ignore" not in c[1]["source"].split("\n", 1)[0], cells)
  for i, cell in cells:
    src_changed = False
    new_src = "\n".join(map(_wrap_command_line, cell["source"].splitlines()))
    try:
      new_src, _ = yapf_api.FormatCode(new_src, style_config=style)
    except yapf_api.errors.YapfError as e:
      se = f"# yapf{e.args[0][9:]}"
      if se not in new_src and tag_errors:
        new_src += "\n" + se
        src_changed = True
    new_src = "\n".join(map(_unwrap_command_line, new_src.splitlines()))
    src_changed = src_changed or cell["source"] != new_src
    if src_changed:
      # pylint: disable=W0125
      d = difflib.unified_diff(cell["source"].splitlines(keepends=True),
                               new_src.splitlines(keepends=True))
      if d:
        diff.append((i + 1, "".join(d)))
        cell["source"] = new_src

  if not diff:
    logger.info("  %s %s", dots, "\u2713")
  elif dry_run:
    for i, d in diff:
      logger.info("[Cell %d]\n%s", i, d)
  else:
    logger.info("  %s %s", dots, "\u21BB")
    with open(filepath, mode="w", encoding="utf-8") as f:
      nbformat.write(notebook, f, version=nbformat.NO_CONVERT)
  return bool(diff)


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  if args.file:
    logger.info("Formatting:")
  for f in args.file:
    format_nb(f,
              dry_run=args.dry_run,
              style=args.style,
              tag_errors=args.tag_errors)
