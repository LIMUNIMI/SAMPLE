"""Evaluation script for BeatsDROP"""
import argparse
import collections
import contextlib
import functools
import glob
import itertools
import io
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
import warnings
from typing import List, Optional, Tuple

import_error = None
try:
  import autorank
  import numpy as np
  import pandas as pd
  import sample
  import sample.beatsdrop.regression  # pylint: disable=W0611
  import tqdm
  from chromatictools import pickle
  from sample import beatsdrop, psycho
  from sample.evaluation import random
  from scipy.io import wavfile
except (ImportError, ModuleNotFoundError) as _import_error:
  import_error = _import_error

logger = logging.getLogger("BeatsDROP-Eval")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the BeatsDROP evaluation script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument(
        "-O",
        "--output",
        metavar="PATH",
        default=None,
        help="Output base path for results",
    )
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )
    self.add_argument(
        "-j",
        "--n-jobs",
        metavar="N",
        default=None,
        type=int,
        help="The number of worker processes to use. "
        "By default, the number returned by os.cpu_count() is used.",
    )
    self.add_argument("-n",
                      "--n-cases",
                      metavar="N",
                      default=1024,
                      type=int,
                      help="The number of tests to perform")
    self.add_argument("--alpha",
                      metavar="P",
                      default=0.05,
                      type=float,
                      help="The threshold for statistical significance")
    self.add_argument("--frequentist",
                      action="store_true",
                      help="Perform frequentist tests (instead of Bayesian)")
    self.add_argument("--tqdm",
                      action="store_true",
                      help="Use tqdm progressbar")
    self.add_argument("--no-resume",
                      dest="resume",
                      action="store_false",
                      help="Do not load previous results, "
                      "but recompute everything")
    self.add_argument("--checkpoint",
                      metavar="PERIOD",
                      default=None,
                      type=int,
                      help="Period for saving checkpoints (the number of "
                      "tests to do before saving another checkpoint)")
    self.add_argument("--wav",
                      metavar="PATH",
                      default=None,
                      help="Folder for writing wav files for test cases")
    self.add_argument("--log-exception",
                      metavar="PATH",
                      default=None,
                      help="Folder for writing logs for test failure, "
                      "instead of raising an exception")
    self.add_argument("--install",
                      action="store_true",
                      help="Install dependencies (no experiment will be run)")

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP evaluation script

    Args:
      argv (tuple): CLI arguments

    Returns:
      Namespace: Parsed arguments"""
    args = self.parse_args(argv)
    setup_logging(args.log_level)
    logger.debug("Args: %s", args)
    return args


def setup_logging(log_level):
  """Setup logger

  Args:
    log_level: Log level"""
  logging.basicConfig(
      level=logging.WARNING,
      format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
  )
  logging.captureWarnings(True)
  logger.setLevel(log_level)


def install_dependencies(*args):
  """Install script dependencies

  Args:
    *args: Extra dependencies to install"""
  logger.info("Updating pip")
  subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"],
                 check=True)
  sample_dir = os.path.join(os.path.dirname(__file__), "..")
  logger.info("Installing module 'sample' and dependencies from folder: %s",
              sample_dir)
  subprocess.run(
      [sys.executable, "-m", "pip", "install", "-U", f"{sample_dir}[plots]"],
      check=True)
  if len(args) > 0:
    logger.info("Installing extra modules: %s", args)
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", *args],
                   check=True)


beat_param_names = ("a0", "a1", "f0", "f1", "d0", "d1", "p0", "p1")
br_param_names = tuple(map("br_".__add__, beat_param_names))
dbr_param_names = tuple(map("dbr_".__add__, beat_param_names))
beatsdrop_eval_result_fields = ("seed", *beat_param_names, *br_param_names,
                                *dbr_param_names)
BeatsDROPEvalResult = collections.namedtuple(
    typename="BeatsDROPEvalResult",
    field_names=beatsdrop_eval_result_fields,
    defaults=itertools.repeat(None, len(beatsdrop_eval_result_fields)))


def test_case(seed: int,
              log_level: Optional[int] = None,
              log_path: Optional[str] = None,
              wav_path: Optional[str] = None) -> Optional[BeatsDROPEvalResult]:
  """BeatsDROP test case

  Args:
    seed (int): RNG seed
    wav_path (str): Path for writing WAV file"""
  # Generate ground truth
  bg = random.BeatsGenerator(onlybeat=True, seed=seed)
  x, fs, ((f0, f1, _), (d0, d1, _), (a0, a1, _), (p0, p1, _)) = bg.audio()
  ground_truth = dict(
      zip(beat_param_names,
          beatsdrop.regression.sort_params((a0, a1, f0, f1, d0, d1, p0, p1))))
  try:
    # Save WAV
    if wav_path is not None:
      wavfile.write(filename=wav_path.format(seed), rate=fs, data=x)
    # Apply SAMPLE
    s = sample.SAMPLE(
        sinusoidal_model__max_n_sines=32,
        sinusoidal_model__reverse=True,
        sinusoidal_model__t=-90,
        sinusoidal_model__save_intermediate=True,
        sinusoidal_model__peak_threshold=-45,
    ).fit(x, sinusoidal_model__fs=fs)
    track = s.sinusoidal_model.tracks_[np.argmax(s.energies_)]
    track_t = np.arange(len(
        track["mag"])) * s.sinusoidal_model.h / s.sinusoidal_model.fs
    track_a = np.flip(track["mag"]) + 6
    track_f = np.flip(track["freq"])

    iok = np.isfinite(track_a)
    track_t = track_t[iok]
    track_a = track_a[iok]
    track_f = track_f[iok]
    # Apply BeatRegression
    br = beatsdrop.regression.BeatRegression().fit(t=track_t,
                                                   a=track_a,
                                                   f=track_f)
    # Apply DualBeatRegression
    dbr = beatsdrop.regression.DualBeatRegression().fit(t=track_t,
                                                        a=track_a,
                                                        f=track_f)

    return BeatsDROPEvalResult(
        seed=seed,
        **ground_truth,
        **dict(zip(br_param_names,
                   beatsdrop.regression.sort_params(br.params_))),
        **dict(
            zip(dbr_param_names,
                beatsdrop.regression.sort_params(dbr.params_))),
    )
  except Exception as e:  # pylint: disable=W0703
    if log_level is not None:
      setup_logging(log_level)
    logger.error("Error in test for seed: %d", seed)
    if log_path is None:
      raise e
    else:
      filename = log_path.format(seed)
      os.makedirs(os.path.dirname(filename), exist_ok=True)
      with open(filename, mode="w", encoding="utf-8") as f:
        f.write(f"Seed: {seed}")
        f.write("\n")
        json.dump(ground_truth, f, indent=2)
        f.write("\n")
        f.write(str(e))
        f.write("\n")
        f.write(traceback.format_exc())


def list2df(results: List[BeatsDROPEvalResult]) -> "pd.DataFrame":
  """Convert a list of results to a pandas dataframe

  Args:
    results (list of BeatsDROPEvalResult): List of results

  Returns:
    DataFrame: DataFrame of results"""
  data = dict(map(lambda k: (k, []), beatsdrop_eval_result_fields))
  for r in filter(lambda r: r.seed is not None, results):
    for k in beatsdrop_eval_result_fields:
      data[k].append(getattr(r, k))
  return pd.DataFrame(data=data)


def prepare_folders(args: argparse.Namespace):
  """Prepare folders for WAV and log files

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace: CLI arguments, augmented"""
  # Make WAV folder
  if args.wav is None:
    args.wav_path = None
  else:
    os.makedirs(args.wav, exist_ok=True)
    args.wav_path = os.path.join(args.wav, "{:.0f}.wav")
  # Logs folder
  args.log_path = None if args.log_exception is None else os.path.join(
      args.log_exception, "{:.0f}.log")
  return args


def load_results(args: argparse.Namespace):
  """Load results from files

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  args.results = list(itertools.repeat(BeatsDROPEvalResult(), args.n_cases))
  seeds = range(args.n_cases)
  args.csv_path = None
  args.ckpt_path = None
  args.last_file = None
  args.last_checkpoint = -1
  args.checkpoints = []
  if args.output is not None:
    args.csv_path = f"{args.output}.csv"
    args.ckpt_path = f"{args.output}.ckpt-{{}}".format
    if args.resume:
      # Find most recent checkpoint
      for fn in glob.glob(args.ckpt_path("*")):  # pylint: disable=W1310
        try:
          pd.read_csv(fn)
        except Exception:  # pylint: disable=W0703
          continue
        args.checkpoints.append(fn)
      if len(args.checkpoints) > 0:
        args.last_checkpoint = max(
            map(lambda s: int(s.rsplit("-", 1)[-1]), args.checkpoints))
        args.last_file = args.ckpt_path(args.last_checkpoint)
      elif os.path.exists(args.csv_path):
        args.last_file = args.csv_path
    if args.resume and args.last_file is not None:
      logger.info("Loading '%s'", args.last_file)
      for _, r in pd.read_csv(args.last_file).iterrows():
        d = {k: r[k] for k in beatsdrop_eval_result_fields}
        d["seed"] = int(d["seed"])
        if d["seed"] < len(args.results):
          args.results[d["seed"]] = BeatsDROPEvalResult(**d)

      def _f(i: int) -> bool:
        """Check that result is not yet computed"""
        return args.results[i].seed is None

      if args.wav_path is None:
        f = _f
      else:

        def f(i: int) -> bool:
          """Also check is WAV file does not exists"""
          return _f(i) or not os.path.exists(args.wav_path.format(i))

      seeds = filter(f, seeds)
  args.seeds = list(seeds)
  return args


def run_tests(args: argparse.Namespace):
  """Run tests

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  if len(args.seeds) == 0:
    logger.info("No tests to do")
  else:
    logger.info("Running tests")
    with mp.Pool(processes=args.n_jobs) as pool:
      it = pool.imap_unordered(
          functools.partial(test_case,
                            log_level=args.log_level,
                            log_path=args.log_path,
                            wav_path=args.wav_path), args.seeds)
      if args.tqdm:
        it = tqdm.tqdm(it, total=len(args.seeds))
      for i, r in enumerate(it):
        if r is not None:
          args.results[r.seed] = r
        # Save checkpoint
        if args.checkpoint is not None and (i + 1) % args.checkpoint == 0:
          args.last_checkpoint += 1
          args.last_file = args.ckpt_path(args.last_checkpoint)
          if args.tqdm:
            it.set_description(f"Checkpoint: '{args.last_file}'")
          else:
            logger.info("Writing checkpoint: '%s'", args.last_file)
          list2df(args.results).to_csv(args.last_file)
  return args


def save_dataframe(args: argparse.Namespace):
  """Save dataframe

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  args.df = list2df(args.results)
  if args.csv_path is not None and (args.csv_path != args.last_file or
                                    not args.resume):
    logger.info("Writing CSV file: '%s'", args.csv_path)
    args.df.to_csv(args.csv_path)
  # Clean checkpoints
  if args.ckpt_path is not None:
    for fn in glob.glob(args.ckpt_path("*")):
      logger.info("Deleting checkpoint: '%s'", fn)
      os.remove(fn)
  return args


def statistical_tests(args: argparse.Namespace):
  """Perform statistical tests on the results

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  residual_premap = {"f": psycho.hz2mel}
  models = {
      "br": "BeatRegression",
      "dbr": "DualBeatRegression",
  }
  variables_to_test = tuple(
      f"{k}{i}" for k, i in itertools.product("fad", range(2)))
  # Load precomputed results
  args.rr_path = None
  if args.output is not None:
    args.rr_path = f"{args.output}_{{}}.dat".format
  args.rank_results = {}
  rr_files = {}
  pops = {}
  for k in variables_to_test:
    pops[k] = {}
    for m, mk in models.items():
      x = args.df[k]
      y = args.df[f"{m}_{k}"]
      if k in residual_premap:
        x = residual_premap[k](x)
        y = residual_premap[k](y)
      c = np.abs(np.subtract(x, y))
      args.df[f"{k}_{m}_ar"] = c
      pops[k][mk] = c
    rr_files[k] = None if args.rr_path is None else args.rr_path(k)
    if args.resume and rr_files[k] is not None and os.path.exists(rr_files[k]):
      logger.info("Loading '%s'", rr_files[k])
      args.rank_results[k] = pickle.read_pickled(rr_files[k])
  # Compute missing results
  non_precomputed = list(
      filter(lambda k: k not in args.rank_results, variables_to_test))
  if len(non_precomputed) > 0:
    it = non_precomputed
    if args.tqdm:
      it = tqdm.tqdm(it)
    with mp.Pool(processes=args.n_jobs) as pool:
      async_results = {}
      for k in non_precomputed:
        async_results[k] = pool.apply_async(
            autorank.autorank,
            kwds=dict(
                data=pd.DataFrame(pops[k]),
                alpha=args.alpha / len(variables_to_test),
                verbose=False,
                order="ascending",
                approach="frequentist" if args.frequentist else "bayesian"))
      for k in it:
        msg = f"Waiting for result '{rr_files[k] or k}'"
        (it.set_description if args.tqdm else logger.info)(msg)
        args.rank_results[k] = async_results[k].get()
        # Save to file
        if rr_files[k] is not None:
          msg = f"Saving result '{rr_files[k]}'"
          (it.set_description if args.tqdm else logger.info)(msg)
          pickle.save_pickled(args.rank_results[k], rr_files[k])
  # Write global report
  def print_report():
    bs = "\\"
    for k in variables_to_test:
      print(f"{bs}section{{{k}}}")
      print(f"{bs}label{{sec:results:{k}}}")
      with io.StringIO() as s:
        with contextlib.redirect_stdout(s):
          autorank.latex_report(args.rank_results[k], complete_document=False)
        print("".join(s.getvalue().splitlines(keepends=True)[2:]))
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autorank.latex_table(args.rank_results[k], label=f"tab:{k}")
      print()

  if args.output is None:
    print_report()
  else:
    args.report_file = f"{args.output}_report.tex"
    logger.info("Writing report file: '%s'", args.report_file)
    with open(args.report_file, mode="w", encoding="utf-8") as f:
      with contextlib.redirect_stdout(f):
        print_report()


def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  if args.install:
    install_dependencies("autorank")
    return
  elif import_error is not None:
    raise import_error
  logger.debug("Preparing folders. Args: %s", args)
  prepare_folders(args)
  logger.debug("Loading results. Args: %s", args)
  load_results(args)
  logger.debug("Running tests. Args: %s", args)
  run_tests(args)
  logger.debug("Saving dataframe. Args: %s", args)
  save_dataframe(args)
  logger.debug("Statistical comparison. Args: %s", args)
  statistical_tests(args)


if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]) or 0)
