"""Widgets for the SAMPLE GUI"""
import logging as _logging


_logging.basicConfig(
  level=_logging.WARNING,
  format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)


logging = _logging.getLogger("SAMPLE-GUI")
