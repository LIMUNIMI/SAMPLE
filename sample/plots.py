"""Plot functions for helping visualization

This module requires extra dependencies, which you can install with

:data:`pip install lim-sample[plots]`"""
import itertools
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=W0611
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
from scipy import signal
from sklearn import base

from sample.beatsdrop import regression as beatsdrop_regression
from sample.sample import SAMPLE
from sample.sms import sm
from sample.utils import dsp as dsp_utils


def _is_beat(m: SAMPLE, track_i: int):
  """Helper function to determine if a track is been detected as beating"""
  return hasattr(
      m, "beat_decisor"
  ) and m.beat_decisor.intermediate.save and m.beat_decisor.intermediate[
      "decision"][track_i]


def sine_tracking_2d(s: Union[sm.SinusoidalModel, SAMPLE], ax=None, **kwargs):
  """Plot sinusoidal tracks detected by the model on two axes,
  one for frequency and one for magnitude

  Args:
    m (sample.sms.sm.SinusoidalModel): Trained sinusoidal
      model (or SAMPLE model)
    ax: Axes list (optional)
    **kwargs: Keyword arguments for :func:`matplotlib.pyplot.plot`

  Returns:
    The axes list"""
  m = s.sinusoidal if isinstance(s, SAMPLE) else s
  if ax is None:
    _, ax = plt.subplots(1, 2, sharex=True)
  tmax = 0
  reverse = getattr(m.tracker, "reverse", False)
  if reverse:
    if m.intermediate.save:
      tmax = len(m.intermediate["stft"]) * m.h / m.fs
    else:
      tmax = max((track["start_frame"] + track["freq"].size) * m.h / m.fs
                 for track in m.tracks_)
  for i, track in enumerate(m.tracks_):
    t_x = (track["start_frame"] + np.arange(track["freq"].size)) * m.h / m.fs
    if reverse:
      t_x = tmax - t_x
    b = _is_beat(s, i)
    ax[0].plot(t_x, track["freq"], "--" if b else "-", **kwargs)
    ax[1].plot(t_x, track["mag"], "--" if b else "-", **kwargs)

  ax[0].grid(zorder=-100)
  ax[0].set_title("frequency")
  ax[0].set_xlabel("time (s)")
  ax[0].set_ylabel("Hz")

  ax[1].grid(zorder=-100)
  ax[1].set_title("magnitude")
  ax[1].set_xlabel("time (s)")
  ax[1].set_ylabel("dB")

  return ax


def sine_tracking_3d(s: Union[sm.SinusoidalModel, SAMPLE], ax=None):
  """Plot sinusoidal tracks detected by the model on one 3D axis

  Args:
    m (sample.sms.sm.SinusoidalModel): Trained sinusoidal
      model (or SAMPLE model)
    ax: 3D axis (optional)

  Returns:
    The 3D axis"""
  m = s.sinusoidal if isinstance(s, SAMPLE) else s
  if ax is None:
    ax = plt.axes(projection="3d")
  for i, track in enumerate(m.tracks_):
    t_x = (track["start_frame"] + np.arange(track["freq"].size)) * m.h / m.fs
    ax.plot(t_x, track["freq"], track["mag"], "--" if _is_beat(s, i) else "-")

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


_TFFunType = Callable[[np.ndarray, float], Tuple[Tuple[float, float],
                                                 np.ndarray]]


def _stft4resynthesis(**kwargs) -> _TFFunType:
  """Define a stft function for use as argument
  of :func:`resynthesis`

  Args:
    **kwargs: Keyword arguments for :func:`scipy.signal.stft`

  Returns:
    callable: Stft function"""

  def _stft4resynthesis_(x, **kws):
    f_stft, _, x_stft = signal.stft(x, **kws, **kwargs)
    return f_stft[[0, -1]], x_stft

  return _stft4resynthesis_


def resynthesis(x: np.ndarray,
                models: Optional[Dict[str, SAMPLE]] = None,
                fs: float = None,
                original: bool = True,
                original_k: str = "Original",
                axs: Optional[Sequence[plt.Axes]] = None,
                fig: Optional[matplotlib.figure.Figure] = None,
                foreach: Optional[Callable[[int, str, np.ndarray], Any]] = None,
                wav_kws: Optional[Dict[str, Any]] = None,
                tf_kws: Optional[Dict[str, Any]] = None,
                tf_fun: _TFFunType = _stft4resynthesis(nperseg=1024),
                db_floor: float = -90,
                **kwargs):
  """Plot a signal and its resynthesis, using different models

  Args:
    x (array): Original signal
    models (dict): Dictionary of :class:`sample.sample.SAMPLE` values
    fs (float): Sampling frequency. If unspecified, it will be inferred
      by the :data:`models`
    original (bool): If :data:`True` (default), also plot original signals
    original_k (str): Label for original signal
    axs (sequence of axes): Axes onto which to plot. If unspecified, they will
      be defined on :data:`fig`
    fig (figure): Figure onto which to put axes. If unspecified,
      the current figure will be used
    foreach (callable): Function to be called for every model. It should take
      three inputs: the model index, the model label and the audio signal
    wav_kws (dict): Additional keyword arguments for wave
      plot (:func:`matplotlib.pyplot.plot`)
    tf_kws (dict): Additional keyword arguments for time-frequency
      plot (:func:`sample.plots.tf_plot`)
    tf_fun (callable): Function for getting a time-frequency representation
      from a signal. It should take an audio array as a positional argument
      and the sampling frequency :data:`fs` as a keyword argument. It should
      return the frequency limits of the representation and the
      time-frequency matrix
    db_floor (float): Lower threshold for tf amplitude values, in dB.
      If :data:`None`, the values will not be converted in dB
    **kwargs: Keyword arguments for :func:`matplotlib.figure.Figure.subplots`

  Returns:
    figure, axes: Figure and list of axes for the plot"""
  if models is None:
    models = {}
  n_models = len(models)
  if fs is None:
    fs = max((m.sinusoidal.fs for m in models.values()), default=44100)
  # Define figure
  if axs is None:
    if fig is None:
      fig = plt.gcf()
    kwargs["sharex"] = kwargs.get("sharex", True)
    kwargs["sharey"] = kwargs.get("sharey", "row")
    axs = fig.subplots(2, n_models + original, squeeze=False, **kwargs)
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
      ax.remove()
    axs = [fig.add_subplot(gs[0, :]), *axs[1, :]]
  elif fig is None:
    fig = axs[0].get_figure()
  wav_kws_ = {"alpha": 0.5, "zorder": 100}
  tf_kws_ = {
      "aspect_ratio": np.diff(axs[1].get_xlim()) / np.diff(axs[1].get_ylim())
  }
  if wav_kws is not None:
    wav_kws_.update(wav_kws)
  if tf_kws is not None:
    tf_kws_.update(tf_kws)

  t = np.arange(x.size) / fs

  wavs = ((original_k, x),) if original else ()
  wavs = itertools.chain(wavs, ((k, m.predict(t)) for k, m in models.items()))
  for i, (k, x_m) in enumerate(wavs):
    axs[0].plot(t, x_m, label=k, **wav_kws_)
    flim, x_tf = tf_fun(x_m, fs=fs)
    if db_floor is not None:
      x_tf = dsp_utils.complex2db(x_tf, floor=db_floor, floor_db=True)
    tf_plot(x_tf, tlim=t[[0, -1]], flim=flim, ax=axs[i + 1], **tf_kws_)
    axs[i + 1].set_title(k)
    if foreach is not None:
      foreach(i, k, x_m)
  axs[0].legend()
  axs[0].grid()
  return fig, axs


def beatsdrop_comparison(
    model: SAMPLE,
    beatsdrops: Dict[str, beatsdrop_regression.BeatRegression],
    x: np.array,
    track_i: int = 0,
    fs: float = None,
    transpose: bool = False,
    signal_hilbert_am: Union[bool, int] = False,
    warnings_ignore: bool = True,
    axs: Optional[Sequence[plt.Axes]] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
):
  """Compare beat regression models

  Args:
    model (SAMPLE): Fitted :class:`sample.sample.SAMPLE` instance
    beatsdrops (dict): Dictionary of
      :class:`sample.beatsdrop.regression.BeatRegression` values
    x (array): Original signal
    track_i (int): Index of the (beating) track of interest
    fs (float): Sampling frequency. If unspecified, it will be inferred
      by the :data:`model`
    transpose (bool): Swap small multiples rows and columns
    signal_hilbert_am (bool or int): If :data:`True`, then plot the Hilbert
      envelope of the signal, instead of the signal. If an :class:`int`,
      then subsample the envelope by this factor
    warnings_ignore (bool): if :data:`True` (default), then ignore warnings
      while fitting beat regression models
    axs (sequence of axes): Axes onto which to plot. If unspecified, they will
      be defined on :data:`fig`
    fig (figure): Figure onto which to put axes. If unspecified,
      the current figure will be used

  Returns:
    figure, axes: Figure and axes for the plot"""
  # Define figure
  if axs is None:
    if fig is None:
      fig = plt.gcf()
    shape = 3, len(beatsdrops)
    if transpose:
      shape = reversed(shape)
    axs = fig.subplots(*shape, sharex=True)
    if transpose:
      axs = axs.T
  else:
    transpose = False
    if fig is None:
      fig = axs[0][0].get_figure()

  if fs is None:
    fs = model.sinusoidal.fs

  _, track_t, track = model._preprocess_track(  # pylint: disable=W0212
      None, x, model.sinusoidal.tracks_[track_i])
  track_a = dsp_utils.db2a(track["mag"])

  # Apply both variants of regression
  with warnings.catch_warnings():
    if warnings_ignore:
      warnings.simplefilter("ignore")
    beatsdrops = {
        k: base.clone(v).fit(t=track_t, a=track["mag"], f=track["freq"])
        for k, v in beatsdrops.items()
    }
  for i, (k, b) in enumerate(beatsdrops.items()):
    am_, a0_, a1_, fm_ = b.predict(track_t, "am", "a0", "a1", "fm")
    fm_ /= 2 * np.pi

    # Amplitude modulation
    if signal_hilbert_am:
      if isinstance(signal_hilbert_am, bool):
        i_env = np.arange(x.size)
      else:
        i_env = (np.arange(np.floor(x.size / signal_hilbert_am).astype(int)) *
                 signal_hilbert_am).astype(int)
      x_a = signal.hilbert(x)
      x_env = np.abs(x_a[i_env])
      axs[0][i].fill_between(i_env / fs,
                             x_env,
                             -x_env,
                             fc="C0",
                             ec="C0",
                             alpha=0.25,
                             label="Signal")
    else:
      axs[0][i].plot(np.arange(x.size) / fs,
                     x,
                     c="C0",
                     alpha=0.25,
                     label="Signal")
    for a, kw in (
        (track_a, {
            "c": "C0",
            "label": "Sinusoidal Track",
            "zorder": 102
        }),
        (am_, {
            "linestyle": "--",
            "c": "C1",
            "label": "Prediction",
            "zorder": 102
        }),
        (a0_, {
            "c": "C3",
            "label": "$A_1(t)$",
            "zorder": 101
        }),
        (a1_, {
            "c": "C4",
            "label": "$A_2(t)$",
            "zorder": 101
        }),
    ):
      a_ = np.copy(a)
      a_[np.less_equal(a, dsp_utils.db2a(-60))] = np.nan
      axs[0][i].plot(track_t, a, **kw)
      axs[1][i].plot(track_t, dsp_utils.a2db(a_), **kw)

    # Frequency modulation
    axs[2][i].plot(track_t,
                   track["freq"],
                   c="C0",
                   zorder=3,
                   label="Sinusoidal Track")
    axs[2][i].plot(track_t, fm_, "--", c="C1", zorder=5, label="Prediction")
    axs[2][i].plot(track_t,
                   np.full_like(track_t, b.params_[2]),
                   c="C3",
                   label=r"$\nu_1$",
                   zorder=4)
    axs[2][i].plot(track_t,
                   np.full_like(track_t, b.params_[3]),
                   c="C4",
                   label=r"$\nu_2$",
                   zorder=4)
    axs[0][i].set_title(k)

  axs[0][0].set_ylabel("amplitude")
  axs[1][0].set_ylabel("amplitude (dB)")
  axs[2][0].set_ylabel("frequency (Hz)")

  for ax in itertools.chain.from_iterable(axs):
    ax.legend(loc="upper right")
    ax.grid()
    yl = ax.get_ylabel()
    if yl:
      yl = ax.set_ylabel(yl)
      yl.set_rotation(0)
      yl.set_horizontalalignment("left")
      yl.set_verticalalignment("bottom")
      ax.yaxis.set_label_coords(-0.05, 1.01)
  for c in range(axs.shape[0]):
    axs[c, -1].set_xlabel("time (s)")
  if transpose:
    axs = axs.T
  return fig, axs
