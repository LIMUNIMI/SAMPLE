"""Utilities for IPython"""
import contextlib
import json
import os
import time
from typing import Optional

import numpy as np
from IPython import display as ipd


class WebAudio(ipd.HTML):
  """Use instead of :class:`IPython.display.Audio` as a workaround for VS Code

  Args:
    x (array): Array of audio samples with shape :data:`(channels, samples)`
      or :data:`(samples,)`
    rate (int): Sample rate
    label (str): Play button label

  Returns:
    IPython.display.HTML: HTML element with WebAudio content"""

  def __init__(self, x, rate: int, label: str = "Play"):
    with open(os.path.join(os.path.dirname(__file__), "web_audio.html"),
              mode="r",
              encoding="utf-8") as f:
      super().__init__(f.read().format(x=json.dumps(
          np.reshape(x, newshape=(1 if np.ndim(x) == 1 else np.shape(x)[0],
                                  -1)).tolist()),
                                       rate=rate,
                                       label=label))


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
