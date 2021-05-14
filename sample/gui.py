"""SAMPLE GUI launcher"""
from chromatictools import cli
from sample.widgets import main


@cli.main(__name__)
def run():
  """Launch the SAMPLE GUI"""
  main.SAMPLERoot().mainloop()
  return 0
