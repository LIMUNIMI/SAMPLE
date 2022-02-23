"""Classes and functions related to psychoacoustic models"""
import numpy as np


def _hz2bark_zwicker(f):
  """Original definition of the Bark scale (Zwicker & Terhardt (1980))

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return 13.0 * np.arctan(7.6e-4 * f) + 3.5 * np.arctan(np.square(f / 7500.0))


def _hz2bark_traunmuller(f):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return (26.81 * f / (1960.0 + f)) - 0.53


def _bark2hz_traunmuller(b):
  """Definition of the Bark scale by Traunmuller
  (Analytical expressions for the tonotopic sensory scale, 1990)

  Args:
    b: Frequency value(s) in Bark

  Returns:
    Frequency value(s) in Hertz"""
  return (0.53 + b) / (26.28 - b) * 1960.0


def _hz2bark_wang(f):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    f: Frequency value(s) in Hertz

  Returns:
    Frequency value(s) in Bark"""
  return 6.0 * np.arcsinh(f / 600.0)


def _bark2hz_wang(b):
  """Definition of the Bark scale by Wang et al. (An objective measure for
  predicting subjective quality of speech coders, 1992)

  Args:
    b: Frequency value(s) in Bark

  Returns:
    Frequency value(s) in Hertz"""
  return 600.0 * np.sinh(b / 6.0)


_hz2bark_dict = dict(
    zwicker=_hz2bark_zwicker,
    traunmuller=_hz2bark_traunmuller,
    wang=_hz2bark_wang,
)


def hz2bark(f, mode: str = "traunmuller"):
  """Convert Hertz to Bark

  Args:
    f: Frequency value(s) in Hertz
    mode (str): Name of the Bark definition (zwicker, traunmuller, or wang)

  Returns:
    Frequency value(s) in Bark"""
  try:
    func = _hz2bark_dict[mode]
  except KeyError as e:
    raise ValueError(
        f"Unsupported Bark mode: '{mode}'. " +
        f"""Supported modes are: {", ".join(f"'{k}'" for k in _hz2bark_dict)}"""
    ) from e
  return func(f)


_bark2hz_dict = dict(
    traunmuller=_bark2hz_traunmuller,
    wang=_bark2hz_wang,
)


def bark2hz(b, mode: str = "traunmuller"):
  """Convert BArk to Hertz

  Args:
    b: Frequency value(s) in Bark
    mode (str): Name of the Bark definition (traunmuller, or wang)

  Returns:
    Frequency value(s) in Hertz"""
  try:
    func = _bark2hz_dict[mode]
  except KeyError as e:
    raise ValueError(
        "Unsupported Bark mode: '{mode}'. " +
        f"""Supported modes are: {", ".join(f"'{k}'" for k in _bark2hz_dict)}"""
    ) from e
  return func(b)
