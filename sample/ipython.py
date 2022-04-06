"""Utilities for IPython"""
from IPython import display as ipd
import numpy as np
import json
import os


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
