"""Benchmarking script for the CIM '22 paper 'SAMPLE: a Python Package
for the Spectral Analysis of Modal Sounds'"""
import argparse
import functools
import itertools
import logging
import operator
import os
import sys
import timeit
from typing import Tuple

import numpy as np
import pandas as pd
import tqdm
from chromatictools import cli
from matplotlib import pyplot as plt

logger = logging.getLogger("CIM22-Performance")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--output",
                      "-O",
                      metavar="PATH",
                      default=None,
                      help="Output file path for results CSV file")
    self.add_argument("--image-output",
                      metavar="PATH",
                      default=None,
                      help="Output file path for image")
    cochleagram_setup_fname = os.path.join(os.path.dirname(__file__),
                                           "cochleagram_setup.py.in")
    self.add_argument("--n-samples",
                      "-n",
                      metavar="N",
                      default=32,
                      type=int,
                      help="Number of samples per case")
    self.add_argument("--n-cases",
                      "-c",
                      metavar="N",
                      default=16,
                      type=int,
                      help="Number of cases per variable")
    self.add_argument("--duration-min",
                      metavar="F",
                      default=0.5,
                      type=float,
                      help="Minimum input duration for testing")
    self.add_argument("--duration-max",
                      metavar="F",
                      default=5.0,
                      type=float,
                      help="Maximum input duration for testing")
    self.add_argument("--stride-min",
                      metavar="F",
                      default=0.0,
                      type=float,
                      help="Minimum stride duration for testing")
    self.add_argument("--stride-max",
                      metavar="F",
                      default=0.010,
                      type=float,
                      help="Maximum stride duration for testing")
    self.add_argument("--fs",
                      metavar="F",
                      default=44100,
                      type=int,
                      help="Sampling frequency")
    self.add_argument("--cochleagram-setup",
                      dest="cochleagram_setup_fname",
                      metavar="PATH",
                      default=cochleagram_setup_fname,
                      help="Formattable script for the cochleagram setup. "
                      f"Default is '{cochleagram_setup_fname}'")
    self.add_argument("--tqdm",
                      action="store_true",
                      help="Use tqdm progressbar")
    mpl_style_fname = os.path.join(os.path.dirname(__file__),
                                   "figures.mplstyle")
    self.add_argument("--mpl-style",
                      metavar="PATH",
                      default=mpl_style_fname,
                      help="Figure style file")
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

    def cochleagram_setup(fs: float = args.fs,
                          stride_time: float = 0,
                          duration: float = 1):
      with open(args.cochleagram_setup_fname, mode="r", encoding="utf-8") as f:
        return f.read().format(fs=fs, stride_time=stride_time, dur=duration)

    args.cochleagram_setup = cochleagram_setup
    return args


def logspace_p(p: np.ndarray,
               start: float,
               stop: float,
               offset: float = 1) -> np.ndarray:
  """Logarithmically spaced samples over the specified interval
  at the specified fractions

  Args:
    p (array): Fractions
    start (float): Starting value
    stop (float): End value
    *args: Positional arguments for :func:`numpy.linspace`
    offset (float): Value to add before the logarithm (to avoid
      logarithm of zero or negative)
    **kwargs: Keyword arguments for :func:`numpy.linspace`

  Returns:
    array: Array of logarithmically spaced samples"""
  lstart, lstop = np.log(np.array((start, stop)) + offset)
  return np.exp(lstart + np.multiply((lstop - lstart), p)) - offset


def logspace(start, stop, *args, offset: float = 1, **kwargs) -> np.ndarray:
  """Logarithmically spaced samples over the specified interval

  Args:
    start (float): Starting value
    stop (float): End value
    *args: Positional arguments for :func:`numpy.linspace`
    offset (float): Value to add before the logarithm (to avoid
      logarithm of zero or negative)
    **kwargs: Keyword arguments for :func:`numpy.linspace`

  Returns:
    array: Array of logarithmically spaced samples"""
  return logspace_p(np.linspace(0, 1, *args, **kwargs),
                    start=start,
                    stop=stop,
                    offset=offset)


# Define convolution statements
convolutions = {
    method: f"fbank.convolve(x, method='{method}')[:, ::stride]"
    for method in ("auto", "overlap-add", "direct", "fft")
}
convolutions["strided"] = "fbank.convolve(x, stride=stride)"


def clean_data(data: dict, args: argparse.Namespace) -> dict:
  """Make sure data length is multiple of number of samples
  and is equal for all columns

  Args:
    data (dict): Columns of data as a :class:`dict`
    args (Namespace): CLI arguments

  Returns:
    dict: Clean data"""
  min_len = (min(map(len, data.values())) // args.n_samples) * args.n_samples
  return {k: v[:min_len] for k, v in data.items()}


def load_data(args: argparse.Namespace) -> dict:
  """Initialize data or load from previous output file

  Args:
    args (Namespace): CLI arguments

  Returns:
    dict: Data"""
  keys = ("method", "stride", "duration")
  keys_i = tuple(f"{k}_i" for k in keys)
  all_keys = (*keys, *keys_i, "timeit")

  if args.output is None or not os.path.exists(args.output):
    return {k: [] for k in all_keys}
  df = pd.read_csv(args.output)
  return clean_data({k: df[k].to_list() for k in all_keys}, args)


def run_benchmarks(data: dict, args: argparse.Namespace) -> dict:
  """Run benchmarks and extend data

  Args:
    data (dict): Columns of data as a :class:`dict`
    args (Namespace): CLI arguments

  Returns:
    dict: Results data"""
  durations = logspace(args.duration_min,
                       args.duration_max,
                       args.n_cases,
                       offset=0,
                       endpoint=True)
  strides = logspace(args.stride_min,
                     args.stride_max,
                     args.n_cases,
                     offset=1,
                     endpoint=True)
  it = itertools.product(
      *tuple(map(enumerate, (durations, strides, convolutions.items()))))
  total_n = functools.reduce(operator.mul,
                             map(len, (durations, strides, convolutions)))
  start_i = len(list(data.values())[0]) // args.n_samples
  it = itertools.islice(it, start_i, total_n)
  if args.tqdm and start_i < total_n:
    it = tqdm.tqdm(it, total=total_n - start_i)
  for (d_i, d), (s_i, s), (k_i, (k, stmt)) in it:
    data["method"].extend(itertools.repeat(k, args.n_samples))
    data["duration"].extend(itertools.repeat(d, args.n_samples))
    data["stride"].extend(itertools.repeat(s, args.n_samples))
    data["method_i"].extend(itertools.repeat(k_i, args.n_samples))
    data["duration_i"].extend(itertools.repeat(d_i, args.n_samples))
    data["stride_i"].extend(itertools.repeat(s_i, args.n_samples))
    data["timeit"].extend(
        timeit.Timer(stmt=stmt,
                     setup=args.cochleagram_setup(duration=d,
                                                  stride_time=s)).repeat(
                                                      number=1,
                                                      repeat=args.n_samples))
  return data


def times_matrix(df: pd.DataFrame,
                 method: str,
                 args: argparse.Namespace,
                 normalize: bool = True) -> np.ndarray:
  """Run benchmarks and extend data

  Args:
    df (DataFrame): Results dataframe
    args (Namespace): CLI arguments
    normalize (bool): Normalize by input duration

  Returns:
    array: Timings matrix"""
  times = np.full((args.n_cases, args.n_cases), np.nan)
  for (stride_i, duration_i), t in df[df["method"] == method].groupby(
      by=["stride_i", "duration_i"])["timeit"]:
    times[duration_i, stride_i] = t.mean()
  if normalize:
    for duration_i, d in df[df["method"] == method].groupby(
        by=["duration_i"])["duration"]:
      times[duration_i, :] /= d.mean()
  return times


def compute_speedup(df: pd.DataFrame,
                    args,
                    target: str = "strided") -> np.ndarray:
  """Compute speed-up w.r.t. the best of all other method

  Args:
    df (DataFrame): Results dataframe
    args (NameSpace): CLI arguments
    target (str): Method to evaluate

  Returns:
    array: Speed-up matrix"""
  best_other = functools.reduce(
      np.minimum,
      (times_matrix(df, k, args) for k in convolutions if k != target))
  return best_other / times_matrix(df, target, args)


def log_speedup_imshow(speedup: np.ndarray,
                       nticks: int = 3,
                       cmap="RdYlGn",
                       origin="lower",
                       **kwargs):
  """Show image color-mapping the logarithm of speed-up

  Args:
    speedup (array): Speed-up matrix
    nticks (int): Number of ticks per side. Total number of ticks will be
      approximately :data:`2*nticks+1`
    **kwargs: Keyword arguments for :func:`matplotlib.pyplot.imshow`

  Returns:
    Image and Colorbar"""
  log_speedup = np.log2(speedup)

  def preprocess_side(positive: bool):
    where = (np.greater if positive else np.less)(log_speedup, 0)
    if where.any():
      m = (np.max if positive else np.min)(log_speedup[where])
      if not positive:
        m *= -1
      log_speedup[where] /= m
      tickstep = max(1, np.round(m / nticks).astype(int))
      ticks = np.arange(1, 1 + nticks) * tickstep
      ticks = ticks[ticks <= m]
      tick_labels = (2**ticks).astype(str)
      ticks = ticks / m
      if not positive:
        ticks = -ticks[::-1]
        bs = "\\"
        tick_labels = list(
            map(lambda s: f"${bs}nicefrac{{1}}{{{s}}}$", tick_labels[::-1]))
      return ticks, tick_labels
    else:
      return [], []

  (neg_t, neg_tl), (pos_t, pos_tl) = list(map(preprocess_side, range(2)))
  ticks = [*neg_t, 0, *pos_t]
  tick_labels = [*neg_tl, "1", *pos_tl]

  im = plt.imshow(log_speedup,
                  vmin=-1,
                  vmax=1,
                  cmap=cmap,
                  origin=origin,
                  **kwargs)
  bar = plt.colorbar(im, ticks=ticks)
  bar.ax.set_yticklabels(tick_labels)
  return im, bar


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  data = load_data(args)

  try:
    run_benchmarks(data, args)
  except KeyboardInterrupt:
    pass
  finally:
    data = clean_data(data, args)
    df = pd.DataFrame(data=data)
    if args.output is not None:
      df.to_csv(args.output)

  log_speedup_imshow(compute_speedup(df, args))

  xticks = plt.xticks()[0][1:-1]
  xticks_p = np.linspace(0, 1, len(xticks))
  xticks = np.round(xticks_p * (args.n_cases - 1)).astype(int)
  xticks_p = xticks / (args.n_cases - 1)
  xtick_values = logspace_p(xticks_p,
                            start=args.stride_min,
                            stop=args.stride_max,
                            offset=1)
  plt.xticks(xticks, list(map(lambda x: f"{x:.1f}", xtick_values * 1000)))

  yticks = plt.yticks()[0][1:-1]
  ytick_values = logspace_p(yticks / (args.n_cases - 1),
                            start=args.duration_min,
                            stop=args.duration_max,
                            offset=0)
  plt.yticks(yticks, list(map(lambda x: f"{x:.0f}", ytick_values * 1000)))

  plt.xlabel("stride (ms)")
  plt.ylabel("duration (ms)")
  # plt.title("Speed-up")

  plt.gcf().set_size_inches(plt.gcf().get_size_inches() * (1, 0.8))
  if args.image_output is None:
    plt.show()
  else:
    plt.savefig(fname=args.image_output)
    plt.clf()
