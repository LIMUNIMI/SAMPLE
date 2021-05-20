"""Audio loading tab"""
from sample.widgets import responsive as tk, logging, utils, pyplot, audio
from tkinter import filedialog, messagebox
from matplotlib import backend_bases
from matplotlib.backends import _backend_tk
import numpy as np
import librosa
import threading
import os


class AudioLoadTab(utils.DataOnRootMixin, tk.Frame):
  """Tab for loading and trimming audio files

  Args:
    args: Positional arguments for :class:`tkinter.ttk.Frame`
    kwargs: Keyword arguments for :class:`tkinter.ttk.Frame`"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.filedialog_dir = os.path.expanduser("~")
    self.filedialog_dir_save = None
    self.filedialog_file = None
    self.responsive(1, 1)

    # --- Pyplot widgets -----------------------------------------------------
    self.plt = pyplot.PyplotFrame(self)
    self.plt.grid(row=0)
    self.ax = self.plt.fig.add_axes((0, 0, 1, 1))

    fc = utils.root_color(self, "TLabel", "background")
    if fc is not None:
      self.ax.set_facecolor(fc)
    self._trim_side = None
    self.plt.canvas.mpl_connect(
      "button_press_event", self.manual_trim_cbk_press
    )
    self.plt.canvas.mpl_connect(
      "button_release_event", self.manual_trim_cbk_release
    )
    # bind checkbutton values to variables to later inspect them
    self._check_vals = dict()
    for k in ("Pan", "Zoom"):
      b = self.plt.toolbar._buttons[k]
      v = tk.BooleanVar(b, value=False)
      self._check_vals[k] = v
      b.config(variable=v)
    # ------------------------------------------------------------------------

    self.bottom_row = tk.Frame(self)
    self.bottom_row.grid(row=1)
    self.bottom_row.responsive(1, 4)

    # Audio load button
    self._plt_lims_valid = False
    self.load_button = tk.Button(self.bottom_row, text="Load")
    self.load_button.grid(column=0, row=0)
    self.load_button.bind("<Button-1>", self.load_cbk)

    # --- Trim start/stop entries --------------------------------------------
    def trim_entry_cbk(tvar, start: bool):
      def trim_entry_cbk_(*args, **kwargs):  # pylint: disable=W0613
        try:
          tval = float(tvar.get())
        except Exception as e:  # pylint: disable=W0703
          logging.warning(
            "Error parsing trim %s value: %s",
            "start" if start else "stop", e
          )
          tval = 0.
        self.set_trim(tval, samples=False, start=start, update=True)
      tvar.trace_add("write", trim_entry_cbk_)
      return trim_entry_cbk_

    self.trim_start_input_var = tk.StringVar(self, value=None)
    self.trim_start_input = tk.Entry(
      self.bottom_row,
      textvariable=self.trim_start_input_var
    )
    trim_entry_cbk(self.trim_start_input_var, True)
    self.trim_start_input.grid(column=1, row=0)

    self.trim_stop_input_var = tk.StringVar(self, value=None)
    self.trim_stop_input = tk.Entry(
      self.bottom_row,
      textvariable=self.trim_stop_input_var
    )
    trim_entry_cbk(self.trim_stop_input_var, False)
    self.trim_stop_input.grid(column=2, row=0)

    _backend_tk.ToolTip.createToolTip(
      self.trim_start_input,
      "Select the start of the region to\n"
      "analyze by typing here the start\n"
      "time or by clicking on the plot"
    )
    _backend_tk.ToolTip.createToolTip(
      self.trim_stop_input,
      "Select the end of the region to\n"
      "analyze by typing here the end\n"
      "time or by clicking on the plot"
    )
    # ------------------------------------------------------------------------

    # Audio play button
    self._tmp_audio = None
    self.play_button = tk.Button(self.bottom_row, text="Play")
    self.play_button.grid(column=3, row=0)
    self.play_button.bind("<Button-1>", self.play_cbk)
    _backend_tk.ToolTip.createToolTip(
      self.play_button,
      "Play back the selected region of audio"
    )

  @property
  def toolbar_on(self) -> bool:
    """It is :data:`True` if any toolbar checkbutton is on"""
    return any(
      v.get()
      for v in self._check_vals.values()
    )

  def play_cbk(self, *args, **kwargs):  # pylint: disable=W0613
    """Audio playback callback"""
    if self.audio_loaded:
      self._tmp_audio = audio.TempAudio(
        self.audio_x[self.audio_trim_start:self.audio_trim_stop],
        self.audio_sr
      )
      self._tmp_audio.play()

  def load_cbk(self, *args, **kwargs):  # pylint: disable=W0613
    """Audio load callback"""
    filename = filedialog.askopenfilename(
      title="Load audio file",
      initialdir=self.filedialog_dir,
      multiple=False,
    )
    if filename:
      logging.info("Loading audio: %s", filename)
      self.filedialog_file = filename
      self.filedialog_dir = os.path.dirname(filename)
      if self.filedialog_dir_save is None:
        self.filedialog_dir_save = self.filedialog_dir
      try:
        x, sr = librosa.load(filename, sr=None, mono=True)
      except Exception as e:  # pylint: disable=W0703
        logging.error("Error loading audio: [%s] %s", type(e).__name__, e)
        messagebox.showerror(
          type(e).__name__,
          str(e) or "".join((
            type(e).__name__, "\n",
            "Filename: ", filename
          ))
        )
      else:
        self.audio_x = x
        self.audio_sr = sr
        self.auto_trim()
        self._plt_lims_valid = False
        self.update_plot()

  def auto_trim(self):
    """Automatically trim audio via onset detection"""
    onsets = librosa.onset.onset_detect(
      self.audio_x, sr=self.audio_sr, units="samples"
    )
    logging.debug("Onsets: %s", onsets)
    onsets = np.array([*onsets, self.audio_x.size])
    onset_max_i = np.argmax(np.diff(onsets)).flatten()[0]
    self.audio_trim_stop = onsets[onset_max_i + 1]
    self.audio_trim_start = onsets[onset_max_i]

  _manual_trim_lock = threading.Lock()

  def set_trim(
    self,
    value,
    start: bool = True,
    samples: bool = True,
    update: bool = False,
    update_vars: bool = False
  ):
    """Set trim index

    Args:
      value (numeric): Trim time or index
      start (bool): If :data:`True` (default), then set the start index,
        otherwise the stop index
      samples (bool): If :data:`True` (default), then :data:`value` is
        interpreted as samples, otherwise in seconds
      update (bool): If :data:`True` (default), then update the plot.
        Default is :data:`False`
      update_vars (bool): If :data:`True` (default), then update the GUI
        entries. Default is :data:`False`"""
    if self.audio_loaded:
      if not samples:
        value = int(value * self.audio_sr)
      if start:
        self.audio_trim_start = min(
          self.audio_trim_stop - 1,
          max(0, value)
        )
        logging.debug("Audio trim start: %d", self.audio_trim_start)
      else:
        self.audio_trim_stop = max(
          self.audio_trim_start + 1,
          min(self.audio_x.size, value)
        )
        logging.debug("Audio trim stop: %d", self.audio_trim_stop)
      if update:
        self.update_plot(update_vars=update_vars)

  def manual_trim_cbk_press(self, event: backend_bases.MouseEvent):
    """Callback for manual trim on canvas press. This sets which index is
    to be set (start or stop) depending on which is closer"""
    with self._manual_trim_lock:
      if self.toolbar_on:
        logging.debug("Press aborted: toolbar on")
        self._trim_side = None
      elif self.audio_loaded and self._trim_side is None:
        x_i = int(event.xdata * self.audio_sr)
        self._trim_side = \
          abs(x_i - self.audio_trim_start) < abs(x_i - self.audio_trim_stop)

  def manual_trim_cbk_release(self, event: backend_bases.MouseEvent):
    """Callback for manual trim on canvas release. This actually
    sets the index value"""
    with self._manual_trim_lock:
      if self.toolbar_on:
        logging.debug("Release aborted: toolbar on")
        self._trim_side = None
      elif self.audio_loaded and self._trim_side is not None:
        self.set_trim(
          event.xdata,
          start=self._trim_side,
          samples=False,
          update=True,
          update_vars=True,
        )
        self._trim_side = None

  def update_plot(self, update_vars: bool = True):
    """Update the audio plot

    Args:
      update_vars (bool): If :data:`True` (default), then
        also update the trim entries"""
    # Save old axis lims to be restored
    if self._plt_lims_valid:
      xlim = self.ax.get_xlim()
      ylim = self.ax.get_ylim()
    else:
      xlim = ylim = (0, 1)
    self.ax.clear()
    if self.audio_x is not None:
      # Plot audio
      t = np.arange(self.audio_x.size) / (self.audio_sr or 1)
      fg = utils.root_color(
        self, "TLabel", "foreground", "C3"  # "#cccccc"
      )
      self.ax.plot(t, self.audio_x, alpha=0.33, c=fg, zorder=5)
      self.ax.grid(True, c=fg, alpha=0.25, zorder=4)
      if self.audio_loaded:
        # Plot audio in trim region
        self.ax.plot(
          t[self.audio_trim_start:self.audio_trim_stop],
          self.audio_x[self.audio_trim_start:self.audio_trim_stop],
          c="C0", alpha=0.5, zorder=6
        )
      if self._plt_lims_valid:
        # Restore old axis lims
        logging.debug("Setting lims: %s-%s", xlim, ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
      else:
        # Invalidate navigation cache
        self.plt.toolbar.update()
      self._plt_lims_valid = True
      if update_vars and self.audio_loaded:
        # Update trim entries
        self.trim_start_input_var.set(
          str(self.audio_trim_start / self.audio_sr)
        )
        self.trim_stop_input_var.set(
          str(self.audio_trim_stop / self.audio_sr)
        )
    else:
      # In no audio is found, reset variables
      self._plt_lims_valid = False
      if update_vars:
        self.trim_start_input_var.set("")
        self.trim_stop_input_var.set("")
    self.plt.canvas.draw()
