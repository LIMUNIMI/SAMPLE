"""Plot figures for the paper 'Acoustic Beats and Where To Find Them:
Theory of Uneven Beats and Applications to Modal Parameters Estimate'"""
import argparse
import functools
import importlib
import logging
import os
import sys
from typing import Callable, Tuple

import matplotlib as mpl
import numpy as np
from chromatictools import cli
from matplotlib import colors
from matplotlib import pyplot as plt

import sample.beatsdrop.regression
import sample.beatsdrop.sample
from sample import beatsdrop, plots, sample
from sample.utils import dsp as dsp_utils

logger = logging.getLogger("BeatsDROP-Figures")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("output", metavar="PATH", help="Output folder path")
    self.add_argument("--ext",
                      metavar="EXT",
                      default=".pdf",
                      help="Output files extension")
    self.add_argument(
        "--saturation",
        metavar="S",
        default=0.75,
        type=float,
        help="Color saturation modifier",
    )
    self.add_argument("--mpl-style",
                      metavar="PATH",
                      default=f"{os.path.splitext(__file__)[0]}.mplstyle",
                      help="Style file")
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )
    for k in self._PLOTS:
      self.add_argument(f"--plot-{k}",
                        dest=k,
                        action="store_true",
                        help=f"Plot the '{k}' plot")

  _PLOTS = {}

  @classmethod
  def register_plot(cls, _name: str, **kwargs):  # pylint: disable=C0103
    """Register plot function to be called via CLI

    Args:
      __name (str): Name of the plot
      **kwargs: Keyword arguments for the plot function"""

    def register_plot_(func: Callable):
      cls._PLOTS[_name] = functools.partial(func, **kwargs)
      return func

    return register_plot_

  def make_plots(self, args: argparse.Namespace):
    """Make all specified plots"""
    for k, f in args.plots.items():
      logger.info("Plot: '%s'", k)
      f(args)

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP figure script

    Args:
      argv (tuple): CLI arguments

    Returns:
      Namespace: Parsed arguments"""
    args = self.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)
    logger.setLevel(args.log_level)
    plots_flags = {
        k: (getattr(args, k, False), f) for k, f in self._PLOTS.items()
    }
    if not any(b for b, _ in plots_flags.values()):
      plots_flags = {k: (True, f) for k, f in self._PLOTS.items()}
    args.plots = {k: f for k, (b, f) in plots_flags.items() if b}
    logger.debug("Args: %s", args)
    plt.style.use(args.mpl_style)
    args.colors = lambda i: resaturate(f"C{i}", args.saturation)
    return args


def save_fig(filename: str,
             args: argparse.Namespace,
             clf: bool = True,
             **kwargs):
  """Save PyPlot figure

  Args:
    filename (str): Figure file name
    args (Namespace): CLI arguments
    clf (bool): If :data:`True` (default), then clear figure after saving
    **kwargs: Keyword arguments for :func:`matplotlib.pyplot.savefig`"""
  filepath = os.path.join(args.output, filename + args.ext)
  logger.info("Saving plot: %s", filepath)
  plt.savefig(fname=filepath, **kwargs)
  if clf:
    plt.clf()


def subplots(vshape: Tuple[int, int] = (1, 1),
             w: float = 1,
             horizontal: bool = False,
             **kwargs):
  """Wrapper for :func:`matplotlib.pyplot.subplots`

  Args:
    vshape (tuple): Shape for vertical plot
    horizontal (bool): Arrange subplots horizontally (columns-first),
      instead of vertically (rows-first)
    w (float): Width multiplier
    **kwargs: Keyword arguments for :func:`matplotlib.pyplot.subplots`"""
  shape = np.flip(vshape) if horizontal else vshape
  if "figsize" not in kwargs:
    kwargs["figsize"] = np.flip(
        shape) * mpl.rcParams["figure.figsize"] * w / shape[1]
  if horizontal:
    share_d = ("col", "row")
    share_d = dict((share_d, np.flip(share_d)))
    for k in ("sharex", "sharey"):
      v = kwargs.get(k, None)
      if isinstance(v, str):
        kwargs[k] = share_d[v]
  fig, axs = plt.subplots(*shape, **kwargs)
  if horizontal and all(d != 1 for d in vshape):
    axs = axs.T
  return fig, axs


def resaturate(c, saturation: float = 1):
  """Modify the color saturation

  Args:
    c (ColorLike): Original color
    saturation (float): Saturation multiplier

  Returns:
    ColorLike: Resaturated color"""
  d = colors.rgb_to_hsv(colors.to_rgb(c))
  d[1] *= saturation
  return colors.hsv_to_rgb(d)


@ArgParser.register_plot("beat", horizontal=True, w=2)
def plot_beat(args, **kwargs):
  """Plot beat pattern

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  _, axs = subplots(vshape=(2, 1), sharex=True, **kwargs)
  b = beatsdrop.Beat(
      a0=0.60,
      a1=0.40,
      f0=9,
      f1=11,
  )
  fs = 44100
  t = np.arange(fs) / fs
  x, am, fm = b.compute(t, ("x", "am", "fm"))
  axs[0].plot(t, x, c=args.colors(2), label="$f(t)$", zorder=12)
  axs[0].plot(t, am, c=args.colors(0), label=r"$2\alpha(t)$", zorder=11)
  axs[0].fill_between(t,
                      am,
                      -am,
                      facecolor=args.colors(0),
                      alpha=0.125,
                      zorder=9)
  axs[0].plot(t,
              np.full_like(t, 2 * b.a_hat(0)),
              "--",
              c=args.colors(4),
              zorder=10,
              label=r"$2\hat{A}$")
  axs[0].plot(t,
              np.full_like(t, 2 * b.a_oln(0)),
              "--",
              c=args.colors(3),
              zorder=10,
              label=r"$2\overline{A}$")

  axs[1].plot(t, fm, c=args.colors(0), label=r"$\omega(t)$", zorder=12)
  axs[1].plot(t,
              np.full_like(t, 2 * np.pi * b.f0(0)),
              "--",
              c=args.colors(4),
              zorder=11,
              label=r"$\omega_1$")
  axs[1].plot(t,
              np.full_like(t, 2 * np.pi * b.f1(0)),
              "--",
              c=args.colors(3),
              zorder=11,
              label=r"$\omega_2$")
  yl = axs[1].set_ylabel(r"angular velocity (rad/s)")
  yl.set_rotation(0)
  yl.set_horizontalalignment("left")
  yl.set_verticalalignment("bottom")
  axs[1].yaxis.set_label_coords(-0.05, 1.01)
  for ax in axs:
    ax.grid()
    ax.legend(loc="lower right").set_zorder(100)
    ax.set_xlabel(r"time (s)")
  if not kwargs.get("horizontal", False):
    axs[0].set_xlabel("")
  axs[0].set_title("Amplitude Modulation")
  axs[1].set_title("Frequency Modulation")
  axs[0].set_xlim(0, 1)
  save_fig("beats", args)
  return args


@ArgParser.register_plot("regression", horizontal=True, w=2)
def plot_regression(args, horizontal: bool = True, **kwargs):
  """Plot regression parameters

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  # --- Synthesize data -------------------------------------------------------
  fs = 44100
  t = np.arange(int(fs * 3)) / fs
  np.random.seed(42)

  a0, a1 = 0.8, 0.2
  f0, f1 = 1100, np.pi
  f0, f1 = f0 + f1, f0 - f1
  d0, d1 = 0.75, 2
  p0, p1 = 0, np.random.rand() * 2 * np.pi

  x = sample.additive_synth(t,
                            freqs=[f0, f1],
                            amps=[a0, a1],
                            decays=[d0, d1],
                            phases=[p0, p1])

  sample_model = sample.SAMPLE(
      sinusoidal__tracker__max_n_sines=32,
      sinusoidal__tracker__reverse=True,
      sinusoidal__t=-90,
      sinusoidal__intermediate__save=True,
      sinusoidal__tracker__peak_threshold=-45,
  ).fit(x, sinusoidal_model__fs=fs)

  #  --- Plot -----------------------------------------------------------------
  fig, axs = subplots(vshape=(3, 2),
                      sharex=True,
                      sharey="row",
                      horizontal=horizontal,
                      **kwargs)
  plots.beatsdrop_comparison(sample_model, {
      "BeatsDROP": beatsdrop.regression.DualBeatRegression(fs=fs),
      "Baseline": beatsdrop.regression.BeatRegression(fs=fs),
  },
                             x,
                             track_i=0,
                             fig=fig,
                             axs=axs,
                             signal_hilbert_am=512,
                             transpose=horizontal)

  save_fig("regression", args)
  # ---------------------------------------------------------------------------
  return args


@ArgParser.register_plot("emd", horizontal=False, ncols=3, w=2)
def plot_emd(args,
             n_points: int = 384,
             ncols: int = 1,
             subq: float = 0.33,
             horizontal: bool = False,
             **kwargs):
  """Plot EMD IMFs

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  emd = importlib.import_module("emd")

  # --- Synthesize data -------------------------------------------------------
  fs = 44100
  t = np.arange(int(fs * 4)) / fs
  np.random.seed(42)

  a0, a1 = 0.8, 0.2
  f0, f1 = 1100, np.pi
  f0, f1 = f0 + f1, f0 - f1
  d0, d1 = 0.75, 2
  p0, p1 = 0, np.random.rand() * 2 * np.pi

  x = sample.additive_synth(t,
                            freqs=[f0, f1],
                            amps=[a0, a1],
                            decays=[d0, d1],
                            phases=[p0, p1],
                            analytical=True)

  # Add noise
  np.random.seed(42)
  a_noise = dsp_utils.db2a(-45)
  x_real = x.real + np.random.randn(np.size(x)) * a_noise

  keys_list = []
  insa_list = []
  imfs_list = []

  # Apply EMD
  logger.debug("Apply EMD")
  imfs = emd.sift.sift(x_real, max_imfs=3)
  _, _, ias = emd.spectra.frequency_transform(imfs, fs, "hilbert")

  keys_list.append("EMD")
  insa_list.append(ias.T)
  imfs_list.append(imfs.T)

  # Apply Ensemble EMD
  logger.debug("Apply Ensemble EMD")
  imfs_e = emd.sift.ensemble_sift(x_real,
                                  max_imfs=3,
                                  nensembles=4,
                                  nprocesses=4)
  _, _, ias_e = emd.spectra.frequency_transform(imfs_e, fs, "hilbert")

  keys_list.append("Ensemble EMD")
  insa_list.append(ias_e.T)
  imfs_list.append(imfs_e.T)

  # Apply Masked EMD
  logger.debug("Apply Masked EMD")
  imfs_m = emd.sift.mask_sift(x_real,
                              max_imfs=3,
                              mask_freqs=(f0 + f1) / (2 * fs))
  _, _, ias_m = emd.spectra.frequency_transform(imfs_m, fs, "hilbert")

  keys_list.append("Masked EMD")
  insa_list.append(ias_m.T)
  imfs_list.append(imfs_m.T)

  # Apply Iterated-Mask EMD
  logger.debug("Apply Iterated-Mask EMD")
  imfs_im = emd.sift.iterated_mask_sift(x_real, max_imfs=3, sample_rate=fs)
  _, _, ias_im = emd.spectra.frequency_transform(imfs_im, fs, "hilbert")

  keys_list.append("Iterated-Mask EMD")
  insa_list.append(ias_im.T)
  imfs_list.append(imfs_im.T)

  # Apply SAMPLE+BeatsDROP
  logger.debug("Apply SAMPLE+BeatsDROP")
  model = beatsdrop.sample.SAMPLEBeatsDROP(
      sinusoidal__tracker__max_n_sines=32,
      sinusoidal__tracker__reverse=True,
      sinusoidal__t=-90,
      sinusoidal__intermediate__save=True,
      sinusoidal__tracker__peak_threshold=-45,
  ).fit(x_real, sinusoidal_model__fs=fs)
  ias_sbd = np.exp(t.reshape((-1, 1)) @ (-2 / model.decays_.reshape(
      (1, -1)))) * model.amps_
  imfs_sbd = ias_sbd * np.cos(
      t.reshape((-1, 1)) @ (2 * np.pi * model.freqs_.reshape((1, -1))))

  keys_list.append("SAMPLE+BeatsDROP")
  insa_list.append(ias_sbd.T)
  imfs_list.append(imfs_sbd.T)

  #  --- Plot -----------------------------------------------------------------
  i_detail = np.arange(n_points)
  subsample_k = np.floor(x_real.size / n_points).astype(int)
  i_subsample = np.multiply(i_detail, subsample_k, dtype=int)
  i_detail = i_detail[:int(n_points * subq)]

  nrows = np.ceil(len(keys_list) / ncols).astype(int)
  _, axs = subplots(vshape=(nrows, ncols),
                    horizontal=horizontal,
                    squeeze=False,
                    **kwargs)
  b = np.mod(np.mod(np.arange(axs.size), ncols), 2)
  s = axs.shape
  axs = axs.reshape((-1,))
  for i in range(len(keys_list) - 1, axs.size - 1):
    b[i] = 2
    axs[i].remove()
    axs[i] = None
  axs = axs.reshape(s)

  plot_k = 0.075

  def _enlarge(a: np.ndarray, k: float = plot_k):
    a = np.array(a)
    return a + np.array([-1, *np.zeros(a.size - 2, dtype=int), 1]).reshape(
        a.shape) * k * np.diff(a.flatten()[[0, -1]])[0]

  yl = _enlarge([-1, 1])
  xl = _enlarge(t[[0, -1]])

  subq_yl = _enlarge(-1 + np.array([0, 2 * subq]), k=[plot_k, -plot_k])
  subq_xl = _enlarge(t[[0, -1]] + [np.diff(t[[0, -1]])[0] * (1 - subq), 0],
                     k=[-plot_k, plot_k])
  t_q = np.linspace(*subq_xl, i_detail.size, endpoint=True)

  dy = np.diff(yl)[0]
  subq_dy = np.diff(subq_yl)[0]

  for k, ax, ias_, imfs_ in zip(
      keys_list, filter(lambda ax: ax is not None, np.ravel(axs)), insa_list,
      imfs_list):
    ax.set_title(k)
    norm_gain = np.max(np.abs(yl)) / np.max(
        np.abs(np.clip([ia[i_detail] for ia in ias_], *yl)))
    for i, (ia, imf) in enumerate(zip(ias_, imfs_)):
      ax.fill_between(t[i_subsample],
                      ia[i_subsample],
                      -ia[i_subsample],
                      ec=args.colors(i),
                      fc=args.colors(i),
                      alpha=0.33,
                      zorder=101)

      # Virtual internal subplot
      ax.fill_between(
          t_q, (np.clip(ia[i_detail] * norm_gain, *yl) - yl[0]) * subq_dy / dy +
          subq_yl[0],
          (np.clip(-ia[i_detail] * norm_gain, *yl) - yl[0]) * subq_dy / dy +
          subq_yl[0],
          ec=args.colors(i),
          fc=args.colors(i),
          alpha=0.33,
          zorder=100)
      ax.plot(t_q,
              (np.clip(imf[i_detail] * norm_gain, *yl) - yl[0]) * subq_dy / dy +
              subq_yl[0],
              c=args.colors(i),
              alpha=0.75,
              zorder=102)

  # Manually handle x-axis labels and tick-labels
  for col in (axs if horizontal else axs.T):
    is_bottom = True
    for ax in reversed(col):
      if ax is None:
        continue
      if is_bottom:
        is_bottom = False
        ax.set_xlabel("time (s)")
      else:
        xt = ax.get_xticks()
        ax.set_xticks(xt, [""] * len(xt))
      ax.set_xlim(xl)

  # Manually handle y-axis labels and tick-labels
  for row in (axs.T if horizontal else axs):
    is_left = True
    for ax in row:
      if ax is None:
        continue
      if is_left:
        is_left = False
        # ax.set_ylabel("amplitude")
      else:
        yt = ax.get_yticks()
        ax.set_yticks(yt, [""] * len(yt))
      ax.set_ylim(yl)
      ax.grid()

  save_fig("emd", args)
  # ---------------------------------------------------------------------------

  return args


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  p = ArgParser(description=__doc__)
  args = p.custom_parse_args(argv)
  logger.debug("Making directory: %s", args.output)
  os.makedirs(args.output, exist_ok=True)
  p.make_plots(args)
