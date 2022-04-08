"""Evaluation script for BeatsDROP"""
import argparse
import functools
import glob
import itertools
import logging
import multiprocessing as mp
import os
import sys
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
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
    return args


BeatsDROPEvalResult_fields = ("seed", "a0", "a1", "f0", "f1", "d0", "d1", "p0",
                              "p1")
BeatsDROPEvalResult = namedtuple(typename="BeatsDROPEvalResult",
                                 field_names=BeatsDROPEvalResult_fields,
                                 defaults=itertools.repeat(
                                     None, len(BeatsDROPEvalResult_fields)))


def test_case(seed: int, wav_path: Optional[str] = None) -> BeatsDROPEvalResult:
  """BeatsDROP test case

  Args:
    seed (int): RNG seed
    wav_path (str): Path for writing WAV file"""
  bg = random.BeatsGenerator(onlybeat=True, seed=seed)
  x, fs, ((f0, f1, _), (d0, d1, _), (a0, a1, _), (p0, p1, _)) = bg.audio()
  a0, a1, f0, f1, d0, d1, p0, p1 = beatsdrop.regression.sort_params(
      (a0, a1, f0, f1, d0, d1, p0, p1))
  if wav_path is not None:
    wavfile.write(filename=wav_path.format(seed), rate=fs, data=x)

  return BeatsDROPEvalResult(
      seed=seed,
      a0=a0,
      a1=a1,
      f0=f0,
      f1=f1,
      d0=d0,
      d1=d1,
      p0=p0,
      p1=p1,
  )


def list2df(results: List[BeatsDROPEvalResult]) -> pd.DataFrame:
  """Convert a list of results to a pandas dataframe

  Args:
    results (list of BeatsDROPEvalResult): List of results

  Returns:
    DataFrame: DataFrame of results"""
  data = dict(map(lambda k: (k, []), BeatsDROPEvalResult_fields))
  for r in filter(lambda r: r.seed is not None, results):
    for k in BeatsDROPEvalResult_fields:
      data[k].append(getattr(r, k))
  return pd.DataFrame(data=data)


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner function"""
  args = ArgParser().custom_parse_args(argv)
  # Make WAV folder
  if args.wav is None:
    wav_path = None
  else:
    os.makedirs(args.wav, exist_ok=True)
    ndigits = np.ceil(np.log10(1 + args.n_cases)).astype(int)
    wav_path = os.path.join(args.wav, f"{{:0{ndigits}.0f}}.wav")
  # Load results
  results = list(itertools.repeat(BeatsDROPEvalResult(), args.n_cases))
  seeds = range(args.n_cases)
  ckpt_path = None
  last_file = None
  last_checkpoint = -1
  if args.resume and args.output is not None:
    # Find most recent checkpoint
    ckpt_path = ".".join((args.output, "ckpt-{}")).format
    checkpoints = glob.glob(ckpt_path("*"))  # pylint: disable=W1310
    if len(checkpoints) > 0:
      last_checkpoint = max(
          map(lambda s: int(s.rsplit("-", 1)[-1]), checkpoints))
      last_file = ckpt_path(last_checkpoint)
    if os.path.exists(args.output):
      last_file = args.output
  if args.resume and last_file is not None:
    for _, r in pd.read_csv(last_file).iterrows():
      d = {k: r[k] for k in BeatsDROPEvalResult_fields}
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
      it = pool.imap_unordered(functools.partial(test_case, wav_path=wav_path),
                               seeds)
      if args.tqdm:
        it = tqdm.tqdm(it, total=len(seeds))
      for i, r in enumerate(it):
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
      for f in glob.glob(ckpt_path("*")):
        logger.info("Deleting checkpoint: '%s'", f)
        os.remove(f)
