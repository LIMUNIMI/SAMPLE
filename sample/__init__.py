"""Implementation of SAMPLE (Spectral Analysis for Modal Parameter Linear
Estimate) and BeatsDROP (Beats Duality for the Resolution Of Partials)

Copyright (c) 2021-2022 Marco Tiraboschi

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
import sys
import types
from typing import Any, Dict, Optional

from chromatictools import cli

from sample.beatsdrop.sample import SAMPLEBeatsDROP
from sample.sample import SAMPLE
import importlib

__version__ = "2.2.0"


@cli.main(__name__)
def _call(logo: Optional[Dict[str, Any]] = None):
  """Print module doc, license and version. Optionally, display the logo"""
  nl = "\n"
  print(f"{__doc__}{nl}{nl}Version: {__version__}")
  if logo is not None:
    importlib.import_module("sample.vid").logo(**logo)


class _CallableModule(types.ModuleType):
  """Make a module callable"""

  def __call__(self, *args, **kwargs):
    return self._call(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
