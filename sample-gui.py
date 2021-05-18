"""SAMPLE GUI stand-alone launcher script

The main use of this script is as an entry-point for pyinstaller"""
from chromatictools import cli
from sample import gui
import sys


cli.main(__name__, *sys.argv[1:])(gui.run)
