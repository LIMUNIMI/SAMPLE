"""SAMPLE GUI stand-alone launcher script

The main use of this script is as an entry-point for pyinstaller"""
from chromatictools import cli
from sample import gui
import multiprocessing
import sys


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Launch the SAMPLE GUI"""
  multiprocessing.freeze_support()
  gui.run(*argv)
