"""SAMPLE GUI launcher"""
from chromatictools import cli
from sample.widgets import main, logging
import sample
import multiprocessing
import logging as _logging
import argparse
import sys


if sys.platform == "linux":
  default_theme = "radiance"
else:
  default_theme = "arc"


def launch(args):
  """Launch the GUI main loop

  Args:
    args (Namespace): Command-line arguments namespace"""
  root = main.main(
    splash_time=args.splash_time,
    gui_kwargs=dict(
      theme=args.theme,
    ),
  )
  root.mainloop()


@cli.main(__name__, *sys.argv[1:])
def run(*argv):
  """Launch the SAMPLE GUI"""
  _logging.basicConfig(
    level=_logging.WARNING,
    format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )
  _logging.captureWarnings(True)

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "-l", "--log-level", dest="log_level", metavar="LEVEL",
    default="INFO", type=lambda s: str(s).upper(),
    help="Set the log level. Default is 'INFO'",
  )
  parser.add_argument(
    "--splash-time", dest="splash_time", metavar="MSEC",
    default=3000, type=int,
    help="Splash time (in milliseconds)",
  )
  parser.add_argument(
    "--theme", metavar="NAME", default=default_theme,
    type=lambda s: str(s).lower(),
    help="GUI theme name (see "
         "https://ttkthemes.readthedocs.io/en/latest/themes.html)",
  )
  args, _ = parser.parse_known_args(argv)

  logging.setLevel(args.log_level)
  logging.info("SAMPLE: version %s", sample.__version__)
  logging.info("Args: %s", args)

  root = multiprocessing.Process(target=launch, args=(args,))
  root.start()
  root.join()
  return 0
