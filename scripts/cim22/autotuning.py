"""Script for the automatic optimization figure for the CIM '22 paper
'SAMPLE: a Python Package for the Spectral Analysis of Modal Sounds'"""
import argparse
import functools
import itertools
import logging
import os
import sys
from typing import Tuple

import matplotlib.collections
import numpy as np
import skopt.plots
import skopt.space
import tqdm
from chromatictools import cli
from matplotlib import pyplot as plt
from sample import optimize
from sample.evaluation import metrics, random
from sample.utils import dsp as dsp_utils
from scipy.io import wavfile

logger = logging.getLogger("CIM22-AutoTuning")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--image-output",
                      metavar="PATH",
                      default=None,
                      help="Output file path for image")
    self.add_argument("--tqdm",
                      action="store_true",
                      help="Use tqdm progressbar")
    mpl_style_fname = os.path.join(os.path.dirname(__file__),
                                   "figures.mplstyle")
    self.add_argument("--seed",
                      metavar="N",
                      default=0,
                      type=lambda s: int(s) if s else None,
                      help="Random number generator seed")
    self.add_argument("--n-triplets",
                      metavar="N",
                      default=16,
                      type=int,
                      help="Number of mode triplets to generate")
    self.add_argument("--wav",
                      metavar="PATH",
                      default=None,
                      help="Output file path for saving test audio")
    self.add_argument("--mpl-style",
                      metavar="PATH",
                      default=mpl_style_fname,
                      help="Figure style file")
    self.add_argument("--n-initial",
                      metavar="N",
                      default=32,
                      type=int,
                      help="Number of exploratory iterations")
    self.add_argument("--n-minimize",
                      metavar="N",
                      default=32,
                      type=int,
                      help="Number of optimization iterations")
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP evaluation script

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
    return args


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  fs = 44100
  # pylint: disable=E1123
  x = np.mean([
      x for x, _, _ in random.BeatsGenerator(
          f_min=60, f_max=12000, seed=args.seed, snr=60).audio(
              fs=fs, size=args.n_triplets)
  ],
              axis=0)
  np.true_divide(x, np.max(np.abs(x)), out=x)

  if args.wav is not None:
    logger.info("Saving test audio: '%s'", args.wav)
    wavfile.write(args.wav, rate=fs, data=x)

  # Define constant arguments
  sample_opt_fixed = {
      "max_n_modes": 64,
      "sinusoidal_model__reverse": True,
      "sinusoidal_model__safe_sine_len": 2,
      "sinusoidal_model__overlap": 0.5,
      "sinusoidal_model__frequency_bounds": (50, 20e3),
  }

  # Define arguments to be optimized
  sample_opt_space = {
      "sinusoidal_model__log_n":
          skopt.space.Integer(6, 14, name="log2(n)"),
      "sinusoidal_model__max_n_sines":
          skopt.space.Integer(16, 128, name="n sines"),
      "sinusoidal_model__peak_threshold":
          skopt.space.Real(-120, -30, name="peak threshold"),
      "sinusoidal_model__min_sine_dur":
          skopt.space.Real(0, 0.5, name="min duration"),
  }

  # Define loss function
  cochleagram_loss = metrics.CochleagramLoss(fs=fs,
                                             normalize=True,
                                             analytical="ir",
                                             stride=int(fs * 0.008),
                                             postprocessing=functools.partial(
                                                 dsp_utils.complex2db,
                                                 floor=-60,
                                                 floor_db=True))

  # Define optimizer
  n_calls = args.n_minimize + args.n_initial
  sample_opt = optimize.SAMPLEOptimizer(
      sample_kw=sample_opt_fixed,
      loss_fn=cochleagram_loss,
      **sample_opt_space,
  )
  tqdm_cbk = optimize.TqdmCallback(
      sample_opt=sample_opt,
      n_calls=n_calls,
      n_initial_points=args.n_initial,
      tqdm_fn=tqdm.tqdm,
  )

  # Optimize
  opt_model, opt_res = sample_opt.gp_minimize(
      x=x,
      fs=fs,
      n_calls=n_calls,
      n_initial_points=args.n_initial,
      callback=tqdm_cbk,
      initial_point_generator="lhs",
      acq_func="LCB",
      random_state=args.seed,
  )

  if args.wav is not None:
    b, e = os.path.splitext(args.wav)
    wav_resynthesis = f"{b}_r{e}"
    logger.info("Saving resynthesis: '%s'", wav_resynthesis)
    x_hat = np.clip(
        opt_model.predict(np.arange(x.size) / fs,
                          phases="random",
                          seed=args.seed), -1, +1)
    wavfile.write(wav_resynthesis, rate=fs, data=x_hat)

  # Plot
  axs = skopt.plots.plot_objective(opt_res, show_points=False)
  axs_1d = []
  axs_2d = []
  map_labels = {"log2(n)": r"$\log_2{n}$"}
  for r, row_axs in enumerate(axs):
    for c, ax in enumerate(row_axs):
      xl = ax.get_xlabel()
      yl = ax.get_ylabel()
      if r == c:
        axs_1d.append(ax)
      elif r > c:
        axs_2d.append(ax)
      if yl in map_labels:
        ax.set_ylabel(map_labels[yl])
      if xl in map_labels:
        ax.set_xlabel(map_labels[xl])
      if xl == "log2(n)":
        ax.set_xticks(np.arange(7, 14, 2))
      if xl == "peak threshold":
        ax.set_xticks(np.arange(-115, -30, 20))

  logger.debug("1D plots: %d", len(axs_1d))
  logger.debug("2D plots: %d", len(axs_2d))
  for ax in axs_1d:
    yt = ax.get_yticks()
    ax.set_yticks(yt, ["" for _ in yt])
    ax.set_ylabel("")
    ax.grid(True)

  cs = itertools.chain.from_iterable(map(lambda ax: ax.get_children(), axs_2d))
  cs = filter(lambda c: isinstance(c, matplotlib.collections.PathCollection),
              cs)
  cs = filter(lambda c: (c.get_fc() != (1, 0, 0, 1)).any(), cs)
  for c in cs:
    c.set_rasterized(True)

  if args.image_output is None:
    plt.show()
  else:
    plt.savefig(fname=args.image_output)
    plt.clf()
