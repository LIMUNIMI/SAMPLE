"""Plot functions for helping visualization

This module requires extra dependencies, which you can install with

:data:`pip install lim-sample[plots]`"""
from sample.sms import sm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d         # pylint: disable=W0611
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611


def sine_tracking_2d(m: sm.SinusoidalModel, ax=None):
  """Plot sinusoidal tracks detected by the model on two axes,
  one for frequency and one for magnitude

  Args:
    m (sample.sms.sm.SinusoidalModel): Trained sinusoidal model
    ax: Axes list (optional)

  Returns:
    The axes list"""
  if ax is None:
    _, ax = plt.subplots(1, 2, sharex=True)
  for track in m.sine_tracker_.all_tracks_:
    t_x = (track["start_frame"] + np.arange(track["freq"].size)) * m.h / m.fs
    ax[0].plot(t_x, track["freq"])
    ax[1].plot(t_x, track["mag"])

  ax[0].grid(zorder=-100)
  ax[0].set_title("frequency")
  ax[0].set_xlabel("time (s)")
  ax[0].set_ylabel("Hz")

  ax[1].grid(zorder=-100)
  ax[1].set_title("magnitude")
  ax[1].set_xlabel("time (s)")
  ax[1].set_ylabel("dB")

  return ax


def sine_tracking_3d(m: sm.SinusoidalModel, ax=None):
  """Plot sinusoidal tracks detected by the model on one 3D axis

  Args:
    m (sample.sms.sm.SinusoidalModel): Trained sinusoidal model
    ax: 3D axis (optional)

  Returns:
    The 3D axis"""
  if ax is None:
    ax = plt.axes(projection="3d")
  for track in m.sine_tracker_.all_tracks_:
    t_x = (track["start_frame"] + np.arange(track["freq"].size)) * m.h / m.fs
    ax.plot(t_x, track["freq"], track["mag"])

  ax.set_xlabel("time (s)")
  ax.set_ylabel("frequency (Hz)")
  ax.set_zlabel("magnitude (dB)")

  return ax
