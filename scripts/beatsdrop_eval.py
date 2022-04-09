"""Evaluation script for BeatsDROP"""
import argparse
import functools
import glob
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sample
import sample.beatsdrop.regression  # pylint: disable=W0611
import tqdm
from chromatictools import cli
from sample import beatsdrop
from sample.evaluation import random
from scipy.io import wavfile

logger = logging.getLogger("BeatsDROP-Eval")


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the BeatsDROP evaluation script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, description: str = __doc__, **kwargs):
    super().__init__(description=description, **kwargs)
    self.add_argument(
        "-O",
        "--output",
        metavar="PATH",
        default=None,
        help="Output path for evaluation CSV file",
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
BeatsDROPEvalResult = namedtuple(typename="BeatsDROPEvalResult",
                                 field_names=beatsdrop_eval_result_fields,
                                 defaults=itertools.repeat(
                                     None, len(beatsdrop_eval_result_fields)))


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


def list2df(results: List[BeatsDROPEvalResult]) -> pd.DataFrame:
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


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner function"""
  args = ArgParser().custom_parse_args(argv)
  ndigits = np.ceil(np.log10(1 + args.n_cases)).astype(int)
  # Make WAV folder
  if args.wav is None:
    wav_path = None
  else:
    os.makedirs(args.wav, exist_ok=True)
    wav_path = os.path.join(args.wav, f"{{:0{ndigits}.0f}}.wav")
  # Logs folder
  log_path = None if args.log_exception is None else os.path.join(
      args.log_exception, f"{{:0{ndigits}.0f}}.log")
  # Load results
  results = list(itertools.repeat(BeatsDROPEvalResult(), args.n_cases))
  seeds = range(args.n_cases)
  ckpt_path = None
  last_file = None
  last_checkpoint = -1
  if args.resume and args.output is not None:
    # Find most recent checkpoint
    ckpt_path = ".".join((args.output, "ckpt-{}")).format
    checkpoints = []
    for fn in glob.glob(ckpt_path("*")):  # pylint: disable=W1310
      try:
        pd.read_csv(fn)
      except Exception:  # pylint: disable=W0703
        continue
      checkpoints.append(fn)
    if len(checkpoints) > 0:
      last_checkpoint = max(
          map(lambda s: int(s.rsplit("-", 1)[-1]), checkpoints))
      last_file = ckpt_path(last_checkpoint)
    if os.path.exists(args.output):
      last_file = args.output
  if args.resume and last_file is not None:
    for _, r in pd.read_csv(last_file).iterrows():
      d = {k: r[k] for k in beatsdrop_eval_result_fields}
      d["seed"] = int(d["seed"])
      results[d["seed"]] = BeatsDROPEvalResult(**d)

    def _f(i: int) -> bool:
      """Check that result is not yet computed"""
      return results[i].seed is None

    if wav_path is None:
      f = _f
    else:

      def f(i: int) -> bool:
        """Also check is WAV file does not exists"""
        return _f(i) or not os.path.exists(wav_path.format(i))

    seeds = filter(f, seeds)
  seeds = list(seeds)
  # Perform tests
  if len(seeds) == 0:
    logger.info("No tests to do")
  else:
    with mp.Pool(processes=args.n_jobs) as pool:
      it = pool.imap_unordered(
          functools.partial(test_case,
                            log_level=args.log_level,
                            log_path=log_path,
                            wav_path=wav_path), seeds)
      if args.tqdm:
        it = tqdm.tqdm(it, total=len(seeds))
      for i, r in enumerate(it):
        if r is not None:
          results[r.seed] = r
        # Save checkpoint
        if args.checkpoint is not None and (i + 1) % args.checkpoint == 0:
          last_checkpoint += 1
          last_file = ckpt_path(last_checkpoint)
          logger.info("Writing checkpoint: '%s'", last_file)
          list2df(results).to_csv(last_file)
  # Save dataframe
  if args.output is not None:
    logger.info("Writing CSV file: '%s'", args.output)
    list2df(results).to_csv(args.output)
    # Clean checkpoints
    if args.checkpoint is not None:
      for fn in glob.glob(ckpt_path("*")):
        logger.info("Deleting checkpoint: '%s'", fn)
        os.remove(fn)
