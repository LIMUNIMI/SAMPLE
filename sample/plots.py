"""Plot functions for helping visualization

This module requires extra dependencies, which you can install with

:data:`pip install lim-sample[plots]`"""
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=W0611
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

from sample.sms import sm


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
  tmax = 0
  if m.reverse:
    if m.save_intermediate:
      tmax = len(m.intermediate_["stft"]) * m.h / m.fs
    else:
      tmax = max((track["start_frame"] + track["freq"].size) * m.h / m.fs
                 for track in m.sine_tracker_.all_tracks_)
  for track in m.sine_tracker_.all_tracks_:
    t_x = (track["start_frame"] + np.arange(track["freq"].size)) * m.h / m.fs
    if m.reverse:
      t_x = tmax - t_x
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


def tf_plot(tf,
            tlim: Tuple[float, float] = (0, 1),
            flim: Tuple[float, float] = (0, 1),
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            ax: Optional[plt.Axes] = None,
            aspect_ratio: float = 4 / 3,
            width: float = 8,
            **kwargs):
  """Plot a time-frequency matrix

  Args:
    tf (matrix): Time-frequency matrix
    tlim (float, float): Extrema for time axis of matrix
    flim (float, float): Extrema for frequency axis of matrix
    xlim (float, float): Extrema for time axis of plot
    ylim (float, float): Extrema for frequency axis of plot
    ax (Axes): Axes onto which to plot the matrix
    aspect_ratio (float): Aspect ratio for image
    width (float): Width for figure size
    **kwargs: Keyword arguments for :func:`imhsow`

  Returns:
    Axes: Axes where the matrix has been plotted"""
  xlim = tlim if xlim is None else xlim
  ylim = flim if ylim is None else ylim
  do_resize = ax is None
  ax = plt.gca() if do_resize else ax

  ax.imshow(tf,
            extent=(*tlim, *flim),
            aspect=np.diff(xlim) / (aspect_ratio * np.diff(ylim)),
            origin="lower",
            **kwargs)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  if do_resize:
    ax.get_figure().set_size_inches(np.array([aspect_ratio, 1]) * width)
  return ax
