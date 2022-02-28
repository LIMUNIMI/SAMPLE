"""Utilities for IPython"""
from IPython import display as ipd
import numpy as np
import json
import os


def WebAudio(x, rate: int, label: str = "Play"):
  """Use instead of :class:`IPython.display.Audio` as a workaround for VS Code

  Args:
    x (array): Array of audio samples with shape :data:`(channels, samples)`
      or :data:`(samples,)`
    rate (int): Sample rate

  Returns:
    IPython.display.HTML: HTML element with WebAudio content"""
  with open(os.path.join(os.path.dirname(__file__), "web_audio.html"),
            mode="r",
            encoding="utf-8") as f:
    return ipd.HTML(f.read().format(x=json.dumps(
        np.reshape(x, newshape=(1 if np.ndim(x) == 1 else np.shape(x)[0],
                                -1)).tolist()),
                                    rate=rate,
                                    label=label))
