"""Plot figures for the paper 'Acoustic Beats and Where To Find Them:
Theory of Uneven Beats and Applications to Modal Parameters Estimate'"""
import argparse
import logging
import os
import sys
from typing import Tuple

import matplotlib as mpl
import numpy as np
from chromatictools import cli
from matplotlib import colors
from matplotlib import pyplot as plt

import sample.beatsdrop.regression
import sample.beatsdrop.sample
from sample import beatsdrop, plots, sample

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


def plot_regression(args, horizontal: bool = True, **kwargs):
  """Plot regression parameters

  Args:
    args (Namespace): CLI arguments

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
                             transpose=horizontal)

  save_fig("regression", args)
  # ---------------------------------------------------------------------------
  return args


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  # Make figure folder
  logger.debug("Making directory: %s", args.output)
  os.makedirs(args.output, exist_ok=True)
  plot_beat(args)
  plot_regression(args, horizontal=True, w=2)
