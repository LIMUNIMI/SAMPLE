"""Evaluation script for the paper 'Acoustic Beats and Where To Find Them:
Theory of Uneven Beats and Applications to Modal Parameters Estimate'"""
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
import matplotlib as mpl
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree
import tqdm
from chromatictools import cli, pickle
from matplotlib import pyplot as plt
from scipy.io import wavfile

import sample
import sample.beatsdrop.regression
import sample.beatsdrop.sample
from sample import beatsdrop, psycho
from sample.evaluation import random
from sample.utils import dsp as dsp_utils

logger = logging.getLogger("BeatsDROP-Eval")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

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
    self.add_argument("--test-decision",
                      action="store_true",
                      help="Test the beat decision rule")
    self.add_argument("--test-fft",
                      metavar="N",
                      default=None,
                      type=int,
                      help="Test the efficacy of increasing the FFT size "
                      "to the specified power-of-two")

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP evaluation script

    Args:
      argv (tuple): CLI arguments

    Returns:
      Namespace: Parsed arguments"""
    args = self.parse_args(argv)
    setup_logging(args.log_level)
    if args.test_decision and args.test_fft is not None:
      raise ValueError(
          "Please, use either '--test-fft' or '--test-decision', not both")
    elif args.test_fft is not None:
      args.fieldnames = beatsdrop_fft_result_fields
      args.result_cls = BeatsDROPFFTResult
      args.test_func = functools.partial(test_case_fft, n=1 << args.test_fft)
      args.output_func = fft_report
    elif args.test_decision:
      args.fieldnames = beatsdrop_decision_result_fields
      args.result_cls = BeatsDROPDecisionResult
      args.test_func = test_case_decision
      args.output_func = decision_report
    else:
      args.fieldnames = beatsdrop_eval_result_fields
      args.result_cls = BeatsDROPEvalResult
      args.test_func = test_case
      args.output_func = statistical_tests
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
three_param_names = ("a0", "a1", "a2", "f0", "f1", "f2", "d0", "d1", "d2", "p0",
                     "p1", "p2")
br_param_names = tuple(map("br_".__add__, beat_param_names))
dbr_param_names = tuple(map("dbr_".__add__, beat_param_names))

beatsdrop_eval_result_fields = ("seed", *beat_param_names, *br_param_names,
                                *dbr_param_names)
BeatsDROPEvalResult = collections.namedtuple(
    typename="BeatsDROPEvalResult",
    field_names=beatsdrop_eval_result_fields,
    defaults=itertools.repeat(None, len(beatsdrop_eval_result_fields)))

beatsdrop_decision_result_fields = ("seed", "beat", "single",
                                    *three_param_names)
BeatsDROPDecisionResult = collections.namedtuple(
    typename="BeatsDROPDecisionResult",
    field_names=beatsdrop_decision_result_fields,
    defaults=itertools.repeat(None, len(beatsdrop_decision_result_fields)))

fft_param_names = tuple(f"{k}_hat" for k in beat_param_names[:6])
beatsdrop_fft_result_fields = ("seed", "count", *beat_param_names,
                               *fft_param_names)
BeatsDROPFFTResult = collections.namedtuple(
    typename="BeatsDROPFFTResult",
    field_names=beatsdrop_fft_result_fields,
    defaults=itertools.repeat(None, len(beatsdrop_fft_result_fields)))

base_model: sample.SAMPLE = sample.SAMPLE(
    sinusoidal__tracker__max_n_sines=32,
    sinusoidal__tracker__reverse=True,
    sinusoidal__t=-90,
    sinusoidal__intermediate__save=True,
    sinusoidal__tracker__peak_threshold=-45,
)


@contextlib.contextmanager
def test_case_context(seed: int,
                      onlybeat: bool = True,
                      log_level: Optional[int] = None,
                      log_path: Optional[str] = None,
                      wav_path: Optional[str] = None,
                      **kwargs):
  """Context manager for test cases"""
  bg = random.BeatsGenerator(seed=seed, onlybeat=onlybeat, **kwargs)
  outs = bg.audio()
  x, fs, ((f0, f1, f2), (d0, d1, d2), (a0, a1, a2), (p0, p1, p2)) = outs
  beat_ground_truth = dict(
      zip(beat_param_names,
          beatsdrop.regression.sort_params((a0, a1, f0, f1, d0, d1, p0, p1))))
  if not onlybeat:
    beat_ground_truth["f2"] = f2
    beat_ground_truth["d2"] = d2
    beat_ground_truth["a2"] = a2
    beat_ground_truth["p2"] = p2
  try:
    # Save WAV
    if wav_path is not None:
      wavfile.write(filename=wav_path.format(seed), rate=fs, data=x)
    # User code
    yield outs, beat_ground_truth
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
        json.dump(beat_ground_truth, f, indent=2)
        f.write("\n")
        f.write(str(e))
        f.write("\n")
        f.write(traceback.format_exc())


def test_case(seed: int,
              log_level: Optional[int] = None,
              log_path: Optional[str] = None,
              wav_path: Optional[str] = None) -> Optional[BeatsDROPEvalResult]:
  """BeatsDROP test case

  Args:
    seed (int): RNG seed
    wav_path (str): Path for writing WAV file"""
  with test_case_context(seed=seed,
                         log_level=log_level,
                         log_path=log_path,
                         wav_path=wav_path,
                         onlybeat=True) as ((x, fs, _), ground_truth):
    # Apply SAMPLE
    s = sklearn.base.clone(base_model).fit(x, sinusoidal__tracker__fs=fs)
    track = s.sinusoidal.tracks_[np.argmax(s.energies_)]
    track_t = np.arange(len(track["mag"])) / s.sinusoidal.frame_rate
    track_a = np.flip(track["mag"]) + 6
    track_f = np.flip(track["freq"])

    iok = np.isfinite(track_a)
    track_t = track_t[iok]
    track_a = track_a[iok]
    track_f = track_f[iok]
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
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


def test_case_decision(
    seed: int,
    log_level: Optional[int] = None,
    log_path: Optional[str] = None,
    wav_path: Optional[str] = None) -> Optional[BeatsDROPEvalResult]:
  """BeatsDROP test case for decision rule

  Args:
    seed (int): RNG seed
    wav_path (str): Path for writing WAV file"""
  s = beatsdrop.sample.SAMPLEBeatsDROP(
      **sklearn.base.clone(base_model).get_params(),
      beat_decisor__intermediate__save=True,
      beat_decisor__th=4,
  )
  with test_case_context(seed=seed,
                         log_level=log_level,
                         log_path=log_path,
                         wav_path=wav_path,
                         onlybeat=False) as ((x, fs, _), ground_truth):
    # Apply SAMPLE
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      s.fit(x, sinusoidal__tracker__fs=fs)
    median_freqs = np.vectorize(lambda track: np.median(track["freq"]))(
        s.sinusoidal.tracks_)

    res = {
        k: s.beat_decisor.intermediate["decision"][np.argmin(
            np.abs(f - median_freqs))] for k, f in {
                "beat": (ground_truth["f0"] + ground_truth["f1"]) / 2,
                "single": ground_truth["f2"],
            }.items()
    }
    return BeatsDROPDecisionResult(seed=seed, **ground_truth, **res)


def test_case_fft(seed: int,
                  n: int,
                  log_level: Optional[int] = None,
                  log_path: Optional[str] = None,
                  wav_path: Optional[str] = None) -> int:
  """BeatsDROP test case for FFT increase

  Args:
    seed (int): RNG seed
    wav_path (str): Path for writing WAV file"""
  s = sklearn.base.clone(base_model)
  s.set_params(sinusoidal__n=n)
  with test_case_context(seed=seed,
                         log_level=log_level,
                         log_path=log_path,
                         wav_path=wav_path,
                         onlybeat=True) as ((x, fs, _), ground_truth):
    # Apply SAMPLE
    s.fit(x, sinusoidal__tracker__fs=fs)

    a_hat = s.amps_
    f_hat = s.freqs_
    d_hat = s.decays_
    if a_hat.size == 0:
      a_hat = f_hat = d_hat = np.zeros(2)
    elif a_hat.size == 1:

      def _rep(a, n: int = 2):
        return np.array([a] * n).flatten()

      a_hat = _rep(a_hat / 2)  # Divide amplitude amongst the two partials
      f_hat = _rep(f_hat)
      d_hat = _rep(d_hat)
    a_hat = a_hat[:2]
    f_hat = f_hat[:2]
    d_hat = d_hat[:2]

    return BeatsDROPFFTResult(
        seed=seed,
        count=len(s.sinusoidal.tracks_),
        **ground_truth,
        **dict(
            zip(
                fft_param_names,
                beatsdrop.regression.sort_params(
                    (*a_hat, *f_hat, *d_hat, 0, 0)))),
    )


def list2df(results: List[BeatsDROPEvalResult], fieldnames) -> "pd.DataFrame":
  """Convert a list of results to a pandas dataframe

  Args:
    results (list of BeatsDROPEvalResult): List of results

  Returns:
    DataFrame: DataFrame of results"""
  data = dict(map(lambda k: (k, []), fieldnames))
  for r in filter(lambda r: r.seed is not None, results):
    for k in fieldnames:
      data[k].append(getattr(r, k))
  return pd.DataFrame(data=data)


def prepare_folders(args: argparse.Namespace):
  """Prepare folders for output, WAV and log files

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace: CLI arguments, augmented"""
  logger.debug("Preparing folders")
  logger.debug("Args: %s", args)
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
  logger.debug("Loading results")
  logger.debug("Args: %s", args)
  args.results = list(itertools.repeat(args.result_cls(), args.n_cases))
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
      for fn in glob.glob(args.ckpt_path("*")):
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
        d = {k: r[k] for k in args.fieldnames}
        d["seed"] = int(d["seed"])
        if d["seed"] < len(args.results):
          args.results[d["seed"]] = args.result_cls(**d)

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
    logger.debug("Args: %s", args)
  else:
    logger.info("Running tests")
    logger.debug("Args: %s", args)
    with mp.Pool(processes=args.n_jobs) as pool:
      it = pool.imap_unordered(
          functools.partial(args.test_func,
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
          list2df(args.results, args.fieldnames).to_csv(args.last_file)
  return args


def save_dataframe(args: argparse.Namespace):
  """Save dataframe

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  logger.debug("Saving results")
  logger.debug("Args: %s", args)
  args.df = list2df(args.results, args.fieldnames)
  if args.csv_path is not None and (args.csv_path != args.last_file or
                                    not args.resume):
    logger.info("Writing CSV file: '%s'", args.csv_path)
    args.df.to_csv(args.csv_path)
  # Clean checkpoints
  if args.ckpt_path is not None:
    for fn in glob.glob(args.ckpt_path("*")):
      logger.debug("Deleting checkpoint: '%s'", fn)
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
  # print(r"\tiny")
  print(r"\centering")
  print(r"\begin{tabular}{c" + "c" * 2 * len(models) + "}")
  print(r"\toprule")
  units = {
      "f": "mel",
      "a": "dB",
      "d": "s",
  }
  syms = {
      "f": "\\nu",
      "a": "a",
      "d": "d",
  }
  for m, mk in models:
    print("& \\multicolumn{2}{c}{", mk, "} %", m)
  print("\\\\")
  for _ in models:
    print("& Med. & $95\\%$ CI", end=" ")
  for k in variables_to_test:
    print("\\\\\n\\midrule")
    print(f"${syms[k[0]]}_{int(k[1:]) + 1}$ ({units[k[0]]}) % {k}")
    for m, mk in models:
      median = rank_result.rankdf["median"][f"{k}_{m}_ar"]
      print(f"& {median:.3f}")
      ci = rank_result.rankdf["CI"][f"{k}_{m}_ar"]
      print(f"& {ci}")
  print(r"\\\bottomrule\\")
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


@contextlib.contextmanager
def print_or_write(path: Optional[str] = None):
  """Print to screen or write to file"""
  if path is None:
    yield
  else:
    logger.info("Writing report file: '%s'", path)
    with open(path, mode="w", encoding="utf-8") as f:
      with contextlib.redirect_stdout(f):
        yield


def statistical_tests(args: argparse.Namespace):
  """Perform statistical tests on the results

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  logger.debug("Statistical comparison")
  logger.debug("Args: %s", args)
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
    kws = {
        "data": pd.DataFrame(pops),
        "alpha": args.alpha,
        "verbose": False,
        "order": "ascending",
        "approach": "frequentist" if args.frequentist else "bayesian"
    }
    if not args.frequentist:
      kws["nsamples"] = args.n_samples
    args.rank_result = autorank.autorank(**kws)
  else:
    logger.info("Reading test results: '%s'", args.rr_path)
    args.rank_result = pickle.read_pickled(args.rr_path)
  if args.rr_path is not None and not os.path.exists(args.rr_path):
    logger.info("Saving test results: '%s'", args.rr_path)
    pickle.save_pickled(args.rank_result, args.rr_path)

  with print_or_write(
      args.output if args.output is None else f"{args.output}_report.tex"):
    print_report(args.rank_result)
  return args


def decision_report(args):
  """Output the classification report for the decision rule

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  logger.info("Decision report")
  logger.debug("Args: %s", args)
  n_samples = args.df.shape[0]
  y_true = np.array(
      [*np.ones(n_samples, dtype=bool), *np.zeros(n_samples, dtype=bool)])
  y_pred = np.array([*args.df["beat"], *args.df["single"]])
  s = sklearn.metrics.classification_report(y_true, y_pred)
  # ---------------------------------------------------------------------------

  # --- Print report ----------------------------------------------------------
  with print_or_write(
      args.output if args.output is None else f"{args.output}_report.txt"):
    print(s)
  # ---------------------------------------------------------------------------

  # --- Augment features ------------------------------------------------------
  df = args.df.copy()
  for k in "afdp":
    for i in range(3):
      df[f"{k}{i}_log"] = np.log(df[f"{k}{i}"])
    for s in ["", "_log"]:
      df[f"{k}{s}_delta"] = np.abs(df[f"{k}0{s}"] - df[f"{k}1{s}"])
      df[f"{k}{s}_avg"] = (df[f"{k}0{s}"] + df[f"{k}1{s}"]) / 2
  features = [
      "seed",
      # "beat",
      # "single",
      "a0",
      "a1",
      "a2",
      "f0",
      "f1",
      "f2",
      "d0",
      "d1",
      "d2",
      # "p0",
      # "p1",
      # "p2",
      # "a0_log",
      # "a1_log",
      # "a2_log",
      "a_delta",
      "a_avg",
      "a_log_delta",
      # "a_log_avg",
      # "f0_log",
      # "f1_log",
      # "f2_log",
      "f_delta",
      "f_avg",
      "f_log_delta",
      # "f_log_avg",
      # "d0_log",
      # "d1_log",
      # "d2_log",
      "d_delta",
      "d_avg",
      "d_log_delta",
      # "d_log_avg",
      # "p0_log",
      # "p1_log",
      # "p2_log",
      "p_delta",
      "p_avg",
      # "p_log_delta",
      # "p_log_avg",
      # "error",
  ]
  # ---------------------------------------------------------------------------

  # --- Feature names ---------------------------------------------------------
  latex_feature_names = {}
  l_keys = {
      "f": "\\nu",
      "p": "\\phi",
      "seed": "\\mathrm{seed}",
  }
  for k in features:
    v = k.split("_", 1)[0]
    pre_v = "".join(filter(str.isalpha, v))
    pre_v = l_keys.get(pre_v, pre_v)
    suf_v = "".join(filter(str.isdigit, v))
    v = f"{pre_v}_{{{int(suf_v) + 1}}}" if suf_v else pre_v
    if "_log" in k:
      v = f"\\log{{{v}}}"
    if "_delta" in k:
      v = f"\\Delta{{{v}}}"
    elif "_avg" in k:
      v = f"\\overline{{{v}}}"
    latex_feature_names[k] = f"${v}$"
  # ---------------------------------------------------------------------------

  # --- Importance ------------------------------------------------------------
  importance_fname = None if args.output is None else f"{args.output}_errors_"\
                                                       "feature_importance.csv"
  targets = ["beat", "single"]
  if importance_fname is None or not os.path.isfile(importance_fname):
    dtc = sklearn.tree.DecisionTreeClassifier(random_state=42)
    n_samples = 1024
    n_folds = 16

    importances = {k: np.empty(n_samples * len(targets)) for k in features}
    importances["target"] = np.empty(n_samples * len(targets), dtype=object)

    rkf = sklearn.model_selection.RepeatedKFold(
        n_splits=n_folds,
        n_repeats=np.ceil(n_samples / n_folds).astype(int),
        random_state=42)
    it = enumerate(itertools.islice(rkf.split(df[features]), n_samples))
    if args.tqdm:
      it = tqdm.tqdm(it, total=n_samples, desc="Decision errors analysis")
      for i, (train_index, _) in it:
        for j, t in enumerate(targets):
          dtc.fit(df[features].iloc[train_index], df[t].iloc[train_index])
          idx = 2 * i + j
          for k, v in zip(dtc.feature_names_in_, dtc.feature_importances_):
            importances[k][idx] = v
          importances["target"][idx] = t
    importances = pd.DataFrame(importances)
    if importance_fname is not None:
      logger.info("Writing report file: '%s'", importance_fname)
      importances.to_csv(importance_fname)
  else:
    importances = pd.read_csv(importance_fname)
  # ---------------------------------------------------------------------------

  rcfile = os.path.join(os.path.dirname(__file__), "figures.mplstyle")
  # --- Plot feature importance -----------------------------------------------
  logger.debug("Plotting feature importance")
  with mpl.rc_context(fname=rcfile):
    _, axs = plt.subplots(
        1,
        2,
        sharex=True,
        figsize=(12, 6),
    )
    for t, ax in zip(targets, axs.flatten()):
      ax.set_title(t)
      ax.grid(axis="x")
      imp_df_t = importances[importances["target"] == t][features]
      sorted_feats = imp_df_t.quantile(
          q=0.05).sort_values().iloc[:].index.to_list()
      ax.boxplot(
          list(map(imp_df_t.__getitem__, sorted_feats)),
          notch=True,
          vert=False,
          flierprops={
              "marker": ".",
              "markersize": 1,
              "alpha": 0.5,
              "zorder": 100,
          },
          patch_artist=True,
          boxprops={
              "linewidth": 1,
              "edgecolor": np.array((0, 0, 0, 0.75)),
              "facecolor": "w",
              "zorder": 101,
          },
          medianprops={
              "color": "C0",
              "zorder": 102,
          },
          showcaps=False,
      )
      ax.set_yticklabels([latex_feature_names.get(k, k) for k in sorted_feats])
    if args.output is None:
      plt.show()
    else:
      fname = f"{args.output}_errors_feature_importance.pdf"
      logger.info("Saving feature importance plot to file: '%s'", fname)
      plt.savefig(fname)
    plt.clf()
  # ---------------------------------------------------------------------------

  # --- Plot histograms -------------------------------------------------------
  if args.output is None:
    hist_dir = None
  else:
    hist_dir = os.path.join(os.path.dirname(args.output), "histograms")
    logger.debug("Making directory: '%s'", hist_dir)
    os.makedirs(hist_dir, exist_ok=True)
  targets_d = {
      "beat": ("Positive", lambda x: x),
      "single": ("Negative", np.logical_not),
  }
  cond_props = itertools.product(targets_d, features)
  if args.tqdm:
    cond_props = tqdm.tqdm(cond_props,
                           desc="Plotting histrograms",
                           total=len(targets_d) * len(features))
  with mpl.rc_context(fname=rcfile):
    _, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
    for t, k in cond_props:
      for ax in axs.flatten():
        ax.cla()
      logger.debug("Plotting histogram of %s conditioned on %s prediction", k,
                   t)
      labels = sorted(df[t].unique(), reverse=True)
      d = [df[df[t] == l][k] for l in labels]
      tk, tkfn = targets_d[t]
      # Histogram
      counts, bins, _ = axs[0].hist(
          d,
          label=tkfn(labels),
          zorder=100,
      )
      axs[0].set_title("Histogram")

      # Posterior
      posterior = counts / np.sum(counts, axis=0)

      bw = np.median(np.diff(bins)) / (len(labels) + 0.5)
      bc = (bins[:-1] + bins[1:]) / 2

      for pi, (tki, p) in enumerate(zip(tkfn(labels), posterior)):
        axs[1].bar(bc + pi * bw, p, label=tki, width=bw, zorder=100)
      axs[1].set_title("Posterior")
      for ax in axs.flatten():
        ax.legend(title=tk)
        ax.grid(axis="y")
        ax.set_xlabel(latex_feature_names.get(k, k))
      if hist_dir is None:
        plt.show()
      else:
        fname = os.path.join(hist_dir, f"{t}_{k}.pdf")
        if args.tqdm:
          cond_props.set_description(fname)
        else:
          logger.debug(
              "Saving histogram of %s conditioned on %s prediction: '%s'", k, t,
              fname)
        plt.savefig(fname)
  # ---------------------------------------------------------------------------

  return args


fft_report_template: str = "Detected at least two tracks in {:.0f}/{:.0f} "\
  "cases ({:.2f}%) using FFT size {:.0f} ({:.2f} seconds at {:.0f} Hz)"


def fft_report(args):
  """Output the performance report for the FFT size increase

  Args:
    args (Namespace): CLI arguments

  Returns:
    Namespace, list: CLI arguments, augmented"""
  logger.info("FFT report")
  logger.debug("Args: %s", args)

  two_tracks = args.df["count"] >= 2
  n = sum(two_tracks)
  t = len(two_tracks)
  nfft = 1 << args.test_fft
  fs = 44100
  s = fft_report_template.format(n, t, n * 100 / t, nfft, nfft / fs, fs)

  with print_or_write(
      args.output if args.output is None else f"{args.output}_report.txt"):
    print(s)
  return args


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  args = ArgParser(description=__doc__).custom_parse_args(argv)
  prepare_folders(args)
  load_results(args)
  run_tests(args)
  save_dataframe(args)
  args.output_func(args)
  logger.info("Done.")
