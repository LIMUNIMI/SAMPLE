"""Utilities for IPython"""
import contextlib
import functools
import itertools
import json
import os
import re
import time
from typing import Any, Dict, Optional

import numpy as np
from IPython import display as ipd

from sample import utils


class WebAudio(ipd.HTML):
  """Use instead of :class:`IPython.display.Audio` as a workaround for VS Code

  Args:
    x (array): Array of audio samples with shape :data:`(channels, samples)`
      or :data:`(samples,)`
    rate (int): Sample rate
    label (str): Play button label"""

  def __init__(self, x, rate: int, label: str = "Play"):
    with open(os.path.join(os.path.dirname(__file__), "web_audio.html"),
              mode="r",
              encoding="utf-8") as f:
      super().__init__(f.read().format(x=json.dumps(
          np.reshape(x, newshape=(1 if np.ndim(x) == 1 else np.shape(x)[0],
                                  -1)).tolist()),
                                       rate=rate,
                                       label=label))


class CollapsibleModelParams(ipd.HTML):
  """Display model parameters in a nested collapsible HTML

  Args:
    model: Model of which the parameters should be displayed"""

  # Regex for sphinx commands
  _SPHINXCLASS_REGEX = re.compile(r"\:\w+\:`.+`")
  _SINGLE_WS_REGEX = re.compile(r"\s")

  @staticmethod
  def _unwrap_sphinx_inner(m: "re.Match"):
    return m.group(0).split("`", 2)[1]

  @staticmethod
  def _unwrap_sphinx(s: str):
    """Remove sphinx commands"""
    return CollapsibleModelParams._SPHINXCLASS_REGEX.sub(
        CollapsibleModelParams._unwrap_sphinx_inner, s)

  @staticmethod
  def _parse_docs(docs: Optional[str] = None):
    """Extract a list of argument descriptions from a docstring"""
    if docs is None:
      return []
    docs = CollapsibleModelParams._unwrap_sphinx(docs)
    docs = docs.split("Args:", 1)
    if len(docs) < 2:
      return []
    docs = filter(lambda s: s, docs[1].split("\n"))
    docs_ = []
    s = []
    indent = np.inf
    for d in docs:
      current_indent = len(
          list(
              itertools.takewhile(
                  CollapsibleModelParams._SINGLE_WS_REGEX.fullmatch, d)))
      if current_indent <= indent:
        indent = current_indent
        if s:
          docs_.append(s)
          s = []
      s.append(d)
    docs_.append(s)
    return list(map(" ".join, map(functools.partial(map, str.strip), docs_)))

  class _ModelWrapper:
    """Meant to wrap the input of :class:`CollapsibleModelParams`"""

    def __init__(self, m):
      self.m = m

    def get_params(self) -> Dict[str, Any]:
      return {type(self.m).__name__: self.m}

  @staticmethod
  def _html_model(model) -> str:
    """Get the HTML body for a model and its arguments"""
    lines = []
    docs = CollapsibleModelParams._parse_docs(model.__doc__)
    for k, v in filter(lambda t: len(t[0].split("__", 1)) == 1,
                       model.get_params().items()):
      is_model = hasattr(v, "get_params")
      if is_model:
        lines.append("<button type=\"button\" class=\"collapsible\">")
      else:
        lines.append("<div class=\"noncollapsible\">")
      rk = re.compile("^" + k + r"\W")
      d = list(filter(rk.match, docs))
      lines.extend(d if d else [k])
      if is_model:
        lines.append("</button>")
        lines.append("<div class=\"content\">")
        lines.append(CollapsibleModelParams._html_model(v))
      lines.append("</div>")
    return "\n".join(lines)

  def __init__(self, model):
    with open(os.path.join(os.path.dirname(__file__),
                           "collapsible_params.html"),
              mode="r",
              encoding="utf-8") as f:
      super().__init__(
          f.read().format(body=self._html_model(self._ModelWrapper(model))))


@contextlib.contextmanager
def time_this(title: Optional[str] = None,
              msg: str = "Execution time: {}",
              unit: str = "ms",
              scale: float = 1000,
              ndigits: int = 0):
  """Time the execution of contextual code and display the
  timing in the IPython display

  Args:
    title (str): Title for the timing message
    msg (str): Timing message, to be formatted with actual timing
    unit (str): Label for unit of measure (should match the :data:`scale`)
    scale (float): Multiplier for timing (should match the :data:`unit`)
    ndigits (int): Number of decimal places to display"""
  start_time = time.time()
  yield start_time
  end_time = time.time()
  dt = (end_time - start_time) * scale
  msg = msg.format(f"{dt:.{ndigits}f} {unit}")
  if title is not None:
    msg = title + msg
  ipd.display(ipd.HTML(msg))


class LabelAndPlayForeach:
  """Class of callables to print label and play audio. To be used in conjuction
  with :func:`sample.plots.resynthesis` as the :data:`foreach` argument."""

  def __init__(self,
               html_kws: Optional[Dict[str, Any]] = None,
               audio_kws: Optional[Dict[str, Any]] = None,
               display_kws: Optional[Dict[str, Any]] = None) -> None:
    self._html_kws = utils.default_kws(html_kws)
    self._audio_kws = utils.default_kws(audio_kws)
    self._display_kws = utils.default_kws(display_kws)
    self._audio_kws["normalize"] = self._audio_kws.get("normalize", False)

  def _label(self,
             i: Optional[int] = None,
             k: Optional[str] = None) -> Optional[ipd.HTML]:
    """Make label widget"""
    if i is None and k is None:
      return None
    if i is None:
      s = k
    elif not k:
      s = str(i + 1)
    else:
      s = f"{i + 1:.0f} - {k}"
    return ipd.HTML(f"<h1>{s}</h1>", **self._html_kws)

  def _audio(self, y: Optional[np.ndarray] = None) -> Optional[ipd.Audio]:
    """Make audio widget"""
    if y is None:
      return None
    if not self._audio_kws["normalize"]:
      y = np.clip(y, -1, 1)
    return ipd.Audio(y, **self._audio_kws)

  def __call__(self,
               i: Optional[int] = None,
               k: Optional[str] = None,
               y: Optional[np.ndarray] = None):
    """Print label and play audio.

    Args:
      i (int): Label index
      k (str): Label text
      y (array): Audio samples"""
    tbd = [
        self._label(i=i, k=k),
        self._audio(y=y),
    ]
    tbd = tuple(filter(lambda w: w is not None, tbd))
    if tbd:
      ipd.display(*tbd, **self._display_kws)
