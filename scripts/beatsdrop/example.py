"""Example of application of SAMPLE+BeatsDROP to a real-world signal"""
import argparse
import io
import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import requests
from chromatictools import cli
from scipy import signal
from scipy.io import wavfile

import sample.beatsdrop.sample
from sample.utils import dsp

logger = logging.getLogger("BeatsDROP-Example")


DEFAULT_SRC = "https://gist.github.com/ChromaticIsobar/" \
              "dcde518ec070b38312ef048f472d92aa/raw/" \
              "3a69a5c6285f4516bae840eb565144772e8809ae/glass.wav"


class ArgParser(argparse.ArgumentParser):
  """Argument parser for the script

  Args:
    **kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--input-url",
                      "-u",
                      default=DEFAULT_SRC,
                      metavar="URL",
                      help="Input audio URL")
    self.add_argument("--input",
                      "-i",
                      default=None,
                      metavar="PATH",
                      help="Input audio path")
    self.add_argument("--cropped",
                      "-c",
                      default=None,
                      metavar="PATH",
                      help="Cropped audio output path")
    self.add_argument("--output",
                      "-o",
                      default=None,
                      metavar="PATH",
                      help="Resynthesis audio output path")
    self.add_argument("--no-crop",
                      action="store_true",
                      help="Do not crop input audio")
    self.add_argument(
        "--crop-start",
        "-s",
        default=None,
        type=float,
        metavar="t",
        help="Starting timestamp for region of interest (in seconds)")
    self.add_argument(
        "--crop-end",
        "-e",
        default=None,
        type=float,
        metavar="t",
        help="Ending timestamp for region of interest (in seconds)")
    self.add_argument(
        "-l",
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        type=lambda s: str(s).upper(),
        help="Set the log level. Default is 'INFO'",
    )

  def custom_parse_args(self, argv: Tuple[str]) -> argparse.Namespace:
    """Customized argument parsing for the BeatsDROP example script

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


def load_audio(args, dtype=np.float64):
  """Load audio from file or URL

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  if args.input is not None:
    args.input_url = None
    logger.debug("Reading from: %s", args.input)
    args.fs, args.x = wavfile.read(args.input)
  else:
    args.input = os.path.basename(args.input_url)
    logger.debug("Downloading from: %s", args.input_url)
    r = requests.get(args.input_url, timeout=60)
    with io.BytesIO() as buf:
      buf.write(r.content)
      del r
      buf.seek(0)
      args.fs, args.x = wavfile.read(buf)
  if np.issubdtype(args.x.dtype, np.integer):
    logger.debug("Converting to %s input of type %s", dtype.__name__,
                 args.x.dtype)
    args.x = np.true_divide(args.x, -np.iinfo(args.x.dtype).min, dtype=dtype)
  if np.ndim(args.x) > 1:
    logger.debug("Converting to mono from %d channels", args.x.shape[1])
    args.x = np.mean(args.x, axis=-1)
  return args


def crop_audio(args):
  """Crop input audio

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  if args.no_crop:
    return args
  i0 = None if args.crop_start is None else int(args.crop_start * args.fs)
  i1 = None if args.crop_end is None else int(args.crop_end * args.fs)
  logger.debug("Cropping between: %s and %s", i0, i1)
  if i0 is None or i1 is None:
    logger.debug("Onset detection")
    ons = np.array([*np.flatnonzero(dsp.onset_detection(args.x)), args.x.size])
    logger.debug("Onsets: %s", ons)
    if i0 is None and i1 is None:
      if ons.size < 2:
        i0 = 0
        i1 = args.x.size
      else:
        i0 = ons[np.argmax(np.diff(ons))]
    if i1 is None:
      i1 = min(ons[np.greater(ons, i0)], default=args.x.size)
    if i0 is None:
      i0 = max(ons[np.less(ons, i1)], default=0)
  else:
    i0 = int(args.crop_start * args.fs)
    i1 = int(args.crop_end * args.fs)
  logger.debug("Cropping between: %s and %s", i0, i1)
  args.x = args.x[i0:i1]
  if args.cropped is None:
    args.cropped = "-cropped".join(os.path.splitext(args.input))
  logger.info("Writing to: %s", args.cropped)
  wavfile.write(args.cropped, args.fs, args.x)
  return args


def fit_model(args, n_jobs: int = 6):
  """Fit model to audio

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  logger.info("Fitting model")
  args.model = sample.beatsdrop.sample.SAMPLEBeatsDROP(
      sinusoidal__tracker__max_n_sines=128,
      sinusoidal__n=1024,
      sinusoidal__w=signal.blackmanharris(1024),
      sinusoidal__tracker__h=64,
      sinusoidal__tracker__frequency_bounds=(50, 20e3),
      sinusoidal__tracker__reverse=True,
      sinusoidal__tracker__min_sine_dur=0.1,
      sinusoidal__tracker__strip_t=0.5,
      sinusoidal__tracker__peak_threshold=-60.0,
      sinusoidal__t=-75.0,
  )
  args.model.fit(args.x, sinusoidal__tracker__fs=args.fs, n_jobs=n_jobs)
  logger.info("Found %d modes", args.model.freqs_.size)
  return args


def resynth_audio(args, n_modes: Optional[int] = None):
  """Resynthesize audio

  Args:
    args (Namespace): CLI arguments
    **kwargs: Keyword arguments for :func:`subplots`

  Returns:
    Namespace: CLI arguments"""
  y = args.model.predict(np.arange(args.x.size) / args.fs, n_modes=n_modes)
  if args.output is None:
    args.output = "-output".join(os.path.splitext(args.input))
  logger.info("Writing to: %s", args.output)
  wavfile.write(args.output, args.fs, y)
  return args


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Script runner"""
  p = ArgParser(description=__doc__)
  args = p.custom_parse_args(argv)
  load_audio(args)
  crop_audio(args)
  fit_model(args)
  resynth_audio(args)
