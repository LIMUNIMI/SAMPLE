"""SAMPLE GUI launcher"""
from chromatictools import cli
from sample.widgets import main, logging
import argparse
import sys


@cli.main(__name__, *sys.argv[1:])
def run(*argv):
  """Launch the SAMPLE GUI"""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "-l", "--log-level", dest="log_level", metavar="LEVEL",
    default="WARNING", type=lambda s: str(s).upper(),
    help="Set the log level. Default is 'WARNING'",
  )
  args, _ = parser.parse_known_args(argv)
  logging.setLevel(args.log_level)
  logging.info("Args: %s", args)

  main.main().mainloop()
  return 0
