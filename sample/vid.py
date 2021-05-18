"""Visual identity"""
from matplotlib import pyplot as plt, patches
import functools
import numpy as np
from typing import Optional


def logo(
  fname: Optional[str] = None,
  decay: float = 44100*0.33,
  frequency: float = 4/44100,
  phase: float = - .75 * np.pi,
  points: int = 44100,
  size_inches: float = 1.,
  tk: float = 1,
  s_0: int = 0,
  s_1: int = 13300,
  clear: bool = True,
  icon: bool = False,
  **kwargs
):
  """Plot the SAMPLE logo

  Args:
    fname (str): File name. If provided, save to file.
    decay (float): Decay of dampended cosine
    frequency (float): Digital frequency of dampended cosine
    phase (float): Initial phase of dampened cosine
    points (int): Number of samples of dampened cosine
    size_inches (float): Size of the figure in inches
    tk (float): Thickness multiplier
    s_0 (int): Starting sample for the :data:`'S'` letter
    s_1 (int): Final sample for the :data:`'S'` letter
    clear (bool): If :data:`True` (default), then clear figure before plotting
    icon (bool): If :data:`True`, then plot the icon
    kwargs: Keyword arguments for :func:`plt.savefig`"""
  if clear:
    plt.clf()
  if icon:
    points //= 2
  time = np.arange(points)
  x = -np.exp(time/(-decay)) * np.cos(
    2 * np.pi * frequency * time + phase
  )
  time = -(time / (points/2) - 1)

  # Make letter 'S'
  thicknesses = (
    int(size_inches * tk),
    *tuple(range(int(size_inches * tk + 2), int(3 * size_inches * tk), 2))
  )
  for i, thick in enumerate(thicknesses):
    plt.plot(
      x[s_0:s_1] - .1, time[s_0:s_1] + 0.1,
      c="C0", zorder=10, linewidth=thick,
      alpha=1 if i == 0 else .66/(len(thicknesses) - 1)
    )

  # Write 'ample'
  if not icon:
    plt.text(0, time[s_1], "ample",
      horizontalalignment="left",
      verticalalignment="baseline",
      zorder=11,
      fontdict=dict(
        family="sans-serif",
        color="w",
        size=7 * size_inches,
      )
    )

  plt.plot(x, time, c="#252525", zorder=5, linewidth=int(size_inches * tk))

  # Black background
  plt.axis("off")
  plt.gca().add_patch(patches.Rectangle(
    (-5, -5), 10, 10,
    linewidth=0, facecolor="black", zorder=1,
  ))

  plt.xlim((-1.15, 1.15))
  plt.ylim(plt.xlim())
  plt.gca().set_aspect(1)
  plt.gcf().set_size_inches(np.full(2, size_inches))

  if fname is not None:
    plt.savefig(fname, **kwargs)
    plt.clf()


icon_plt_fn = functools.partial(
  logo, format="png",
  bbox_inches="tight", pad_inches=0,
  size_inches=1, icon=True, tk=5,
)


logo_plt_fn = functools.partial(
  logo, format="png", size_inches=8,
  bbox_inches="tight", pad_inches=0,
)
