"""Evaluation script for BeatsDROP"""
import argparse
import collections
import contextlib
import functools
import glob
import io
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
import warnings
from typing import List, Optional, Tuple

import autorank
import numpy as np
import pandas as pd
import sample
import sample.beatsdrop.regression  # pylint: disable=W0611
import tqdm
from chromatictools import pickle
from sample import beatsdrop, psycho
from sample.evaluation import random
from sample.utils import dsp as dsp_utils
from scipy.io import wavfile

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
    self.add_argument(
        "--n-samples",
        default=1024,  # 50000
        type=int,
        help="Number of samples used to estimate the posterior "
        "probabilities with the Bayesian signed rank test")
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
  """Prepare folders for output, WAV and log files

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace: CLI arguments, augmented"""
  # Make output file folder
  if args.output is not None:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
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


def print_report(rank_result):
  """Prints report for autorank result

  Args:
    rank_result (RankResult): Result from :func:`autorank.autorank`"""
  models = (
      ("dbr", "BeatsDROP"),
      ("br", "Baseline"),
  )
  variables_to_test = tuple(
      f"{k}{i}" for k, i in itertools.product("fad", range(2)))
  with io.StringIO() as s:
    with contextlib.redirect_stdout(s):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autorank.latex_report(rank_result,
                              complete_document=True,
                              generate_plots=False)
    s = s.getvalue().splitlines(keepends=True)
  s.insert(1, r"\usepackage[margin=0.75in]{geometry}")
  print("".join(s[:-1]))

  print(r"\newpage")

  frequentist = rank_result.decision_matrix is None
  if frequentist:
    # Frequentist
    def decision(k1, k2):
      """Get the answer to the question: 'How is k1 wrt k2?'"""
      rd = np.squeeze(np.diff(rank_result.rankdf["meanrank"][[k1, k2]]))
      if abs(rd) < rank_result.cd:
        return "equal"
      elif rd > 0:
        return "smaller"
      else:
        return "larger"
  else:
    # Bayesian
    def decision(k1, k2):
      """Get the answer to the question: 'How is k1 wrt k2?'"""
      decision = rank_result.decision_matrix[k1][k2]
      if not isinstance(decision, str) and np.isnan(decision):
        decision = rank_result.decision_matrix[k2][k1]
        if decision == "smaller":
          decision = "larger"
        elif decision == "larger":
          decision = "smaller"
      return decision

  # Custom table
  print(r"\begin{table*}[ht]")
  print(r"\tiny")
  print(r"\centering")
  print(r"\begin{tabular}{llcccccc}")
  print(r"\toprule")
  print("&", end="")
  units = {
      "f": "mel",
      "a": "dB",
      "d": "s",
  }
  syms = {
      "f": r"\nu",
      "a": r"A^0",
      "d": "d",
  }
  print("&")
  for i, k in enumerate(variables_to_test):
    print(f"${syms[k[0]]}_{int(k[1:]) + 1}$ ({units[k[0]]})",
          r"\\" if i == len(variables_to_test) - 1 else "&", "%", k)
  print(r"\\")
  for m, mk in models:
    print(r"\midrule")
    print(mk, "& Median &")
    for i, k in enumerate(variables_to_test):
      median = rank_result.rankdf["median"][f"{k}_{m}_ar"]
      print(f"{median:.3f}", r"\\" if i == len(variables_to_test) - 1 else "&",
            "%", k)
    if frequentist:
      cip = ""
    else:
      cip = f"${(1 - rank_result.alpha) * 100:.0f}" + r"\%$ "
    print(f"& {cip}CI &")
    for i, k in enumerate(variables_to_test):
      ci = rank_result.rankdf["CI"][f"{k}_{m}_ar"]
      print(ci, r"\\" if i == len(variables_to_test) - 1 else "&", "%", k)
  print(r"\midrule")
  print(r"& Best &")
  for i, k in enumerate(variables_to_test):
    ks = tuple(f"{k}_{m}_ar" for m, _ in models)
    best = decision(ks[0], ks[1])
    best = dict(zip(("smaller", "larger"),
                    models)).get(best, (best, r"\textit{" + best + "}"))[1]
    print(best, r"\\" if i == len(variables_to_test) - 1 else "&", "%", k)
  print(r"\bottomrule\\")
  print(r"\end{tabular}")
  print(r"\caption{Summary of test results}")
  print(r"\label{tab:results}")
  print(r"\end{table*}")

  print("Comparing the two models:")
  print(r"\begin{itemize}")
  for k in variables_to_test:
    print(r"\item{", k, "} median error of br is",
          decision(f"{k}_br_ar", f"{k}_dbr_ar"), "wrt dbr")
  print(r"\end{itemize}")

  print("\nComparing the two partials:")
  print(r"\begin{itemize}")
  for k in "fad":
    print(r"\item{", k, "}")
    print(r"\begin{itemize}")
    for m, _ in models:
      print(r"\item{", m, "} median error on", f"{k}0", "is",
            decision(f"{k}0_{m}_ar", f"{k}1_{m}_ar"), "wrt", f"{k}1")
    print(r"\end{itemize}")
  print(r"\end{itemize}")

  print(s[-1])


def statistical_tests(args: argparse.Namespace):
  """Perform statistical tests on the results

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  residual_premap = {
      "f": psycho.hz2mel,
      "a": dsp_utils.a2db,
  }
  models = (
      "br",
      "dbr",
  )
  variables_to_test = tuple(
      f"{k}{i}" for k, i in itertools.product("fad", range(2)))
  # Load precomputed results
  args.rr_path = None
  if args.output is not None:
    args.rr_path = f"{args.output}_rankresult.dat"
  pops = {}
  for k in variables_to_test:
    for m in models:
      x = args.df[k]
      y = args.df[f"{m}_{k}"]
      if k[0] in residual_premap:
        x = residual_premap[k[0]](x)
        y = residual_premap[k[0]](y)
      c = np.abs(np.subtract(x, y))
      kmk = f"{k}_{m}_ar"
      args.df[kmk] = c
      pops[kmk] = c
  if args.rr_path is None or not os.path.exists(args.rr_path):
    logger.info("Running statistical tests")
    kws = dict(data=pd.DataFrame(pops),
               alpha=args.alpha,
               verbose=False,
               order="ascending",
               approach="frequentist" if args.frequentist else "bayesian")
    if not args.frequentist:
      kws["nsamples"] = args.n_samples
    args.rank_result = autorank.autorank(**kws)
  else:
    logger.info("Reading test results: %s", args.rr_path)
    args.rank_result = pickle.read_pickled(args.rr_path)
  if args.rr_path is not None and not os.path.exists(args.rr_path):
    logger.info("Saving test results: %s", args.rr_path)
    pickle.save_pickled(args.rank_result, args.rr_path)

  if args.output is None:
    print_report(args.rank_result)
  else:
    args.report_file = f"{args.output}_report.tex"
    logger.info("Writing report file: '%s'", args.report_file)
    with open(args.report_file, mode="w", encoding="utf-8") as f:
      with contextlib.redirect_stdout(f):
        print_report(args.rank_result)


def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
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
  logger.info("Done.")


if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]) or 0)
