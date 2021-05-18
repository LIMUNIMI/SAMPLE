"""Analysis tab"""
from sample.widgets import responsive as tk, pyplot, utils, audio, logging
from sample import plots
from tkinter import messagebox, filedialog
import numpy as np
import json
import os


class AnalysisTab(utils.DataOnRootMixin, tk.Frame):
  """Tab for SAMPLE analysis

  Args:
    pad_top_w (float): Padding for top axes width (as a fraction of the
      whole figure)
    pad_top_h (float): Padding for top axes height (as a fraction of the
      whole figure)
    pad_bottom_w (float): Padding for bottom axis width (as a fraction
      of the whole figure)
    pad_bottom_h (float): Padding for bottom axis height (as a fraction
      of the whole figure)
    args: Positional arguments for :class:`tkinter.ttk.Frame`
    kwargs: Keyword arguments for :class:`tkinter.ttk.Frame`"""
  def __init__(
    self,
    *args,
    pad_top_w: float = 0.09,
    pad_top_wm: float = 0.02,
    pad_top_h: float = 0.02,
    pad_bottom_w: float = 0.05,
    pad_bottom_h: float = 0.05,
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.filedialog_dir_save = None
    self.responsive(1, 1)

    # --- Pyplot widgets -----------------------------------------------------
    self.plt = pyplot.PyplotFrame(self)
    self.plt.grid(row=0)

    top_width = 0.5 - pad_top_w - 0.5 * pad_top_wm
    top_height = 0.5 - 2 * pad_top_h
    self.ax = (
      self.plt.fig.add_axes((
        pad_top_w,
        0.5 + pad_top_h,
        top_width,
        top_height,
      )),
      self.plt.fig.add_axes((
        0.50 + pad_top_wm / 2,
        0.50 + pad_top_h,
        top_width,
        top_height,
      )),
      self.plt.fig.add_axes((
        pad_bottom_w,
        pad_bottom_h,
        1 - 2 * pad_bottom_w,
        0.5 - 2 * pad_bottom_h,
      )),
    )
    fc = utils.root_color(self, "TLabel", "background")
    if fc is not None:
      self.plt.fig.set_facecolor(fc)
    for ax in self.ax:
      ax.set_frame_on(False)
      ax.set_xticks(())
      ax.set_yticks(())
    # ------------------------------------------------------------------------

    self.bottom_row = tk.Frame(self)
    self.bottom_row.grid(row=1)
    self.bottom_row.responsive(1, 4)

    # Analysis button
    self.analysis_button = tk.Button(self.bottom_row, text="Analyze")
    self.analysis_button.bind("<Button-1>", self.analysis_cbk)
    self.analysis_button.grid(column=0, row=0)

    # Audio play buttons
    self._tmp_audio = None
    self.play_button_o = tk.Button(self.bottom_row, text="Play Original")
    self.play_button_o.grid(column=1, row=0)
    self.play_button_o.bind("<Button-1>", self.play_cbk(True))
    self.play_button_r = tk.Button(self.bottom_row, text="Play Resynthesis")
    self.play_button_r.grid(column=2, row=0)
    self.play_button_r.bind("<Button-1>", self.play_cbk(False))

    # Export button
    self.export_button = tk.Button(self.bottom_row, text="Export")
    self.export_button.bind("<Button-1>", self.export_cbk)
    self.export_button.grid(column=3, row=0)

  def update_plot(self):
    """Update analysis and resynthesis figure"""
    for ax in self.ax:
      ax.clear()
    m = self.sample_object.sinusoidal_model
    stft = np.array([mx for mx, _ in m.intermediate_["stft"]]).T
    tmax = max((
        track["start_frame"] + track["freq"].size
        for track in m.sine_tracker_.all_tracks_
      ), default=0,
    ) * m.h / m.fs

    plots.sine_tracking_2d(m, ax=self.ax)

    if tmax > 0:
      xlim = (0, tmax)
    else:
      xlim = self.ax[0].get_xlim()
    ylim = self.ax[0].get_ylim()

    self.ax[0].imshow(
      stft, cmap="Greys",
      origin="lower",  aspect="auto",
      extent=(*xlim, 0, m.fs/2),
    )
    self.ax[0].set_ylim(ylim)
    self.ax[0].set_xlim(xlim)
    self.ax[0].grid(False)
    self.ax[0].set_title("")
    self.ax[0].set_ylabel("frequency (Hz)")

    self.ax[1].set_title("")
    self.ax[1].set_ylabel("magnitude (dB)")
    self.ax[1].yaxis.tick_right()
    self.ax[1].yaxis.set_label_position("right")

    x = self.audio_x[self.audio_trim_start:self.audio_trim_stop]
    t = np.arange(x.size) / self.audio_sr
    if self.audio_resynth_x is None:
      self.audio_resynth_x = np.clip(
        self.sample_object.predict(t), -1, +1
      )
    x_hat = self.audio_resynth_x
    self.ax[2].plot(
      t, x, c="C0", alpha=0.5, zorder=6, label="original",
    )
    self.ax[2].plot(
      t, x_hat, c="C1", alpha=0.5, zorder=6, label="resynthesis",
    )

    fc = utils.root_color(self, "TLabel", "foreground", key="labelcolor")
    self.ax[2].legend(
      loc="lower right", **fc,
      **utils.root_color(self, "TLabel", "background", key="facecolor"),
    )

    self.ax[2].set_xticks(())
    self.ax[2].set_yticks(())
    if len(fc) > 0:
      fc = next(iter(fc.values()))
      for ax in self.ax[:2]:
        ax.tick_params(axis="both", colors=fc)
        ax.spines["left"].set_color(fc)
        ax.spines["bottom"].set_color(fc)
        ax.xaxis.label.set_color(fc)
        ax.yaxis.label.set_color(fc)

    self.plt.canvas.draw()

  def analysis_cbk(self, *args, **kwargs):  # pylint: disable=W0613
    """Analysis callback"""
    if not self.audio_loaded:
      messagebox.showerror(
        "No audio",
        "You have to load an audio file before analyzing"
      )
      return
    x = self.audio_x[self.audio_trim_start:self.audio_trim_stop]
    try:
      self.sample_object.fit(
        x, sinusoidal_model__fs=self.audio_sr,
        sinusoidal_model__save_intermediate=True,
      )
    except Exception as e:  # pylint: disable=W0703
      messagebox.showerror(
        type(e).__name__, str(e)
      )
      return
    messagebox.showinfo("Success", "Analysis done!")
    self.audio_resynth_x = None
    self.update_plot()

  def play_cbk(self, original: bool = True):
    """Audio playback callback constructor"""
    def play_cbk_(*args, **kwargs):  # pylint: disable=W0613
      """Audio playback callback"""
      if not self.audio_loaded:
        messagebox.showerror(
          "No audio",
          "You have to load an audio file before playing anything back"
        )
        return
      x = self.audio_x[self.audio_trim_start:self.audio_trim_stop]
      if not original:
        if self.audio_resynth_x is None:
          try:
            self.audio_resynth_x = np.clip(self.sample_object.predict(
              np.arange(x.size) / self.audio_sr,
            ), -1, +1)
          except AttributeError:
            messagebox.showerror(
              "Not analyzed",
              "You have to analyse an audio file before "
              "playing back the resynthesis"
            )
            self.audio_resynth_x = None
            return
        x = self.audio_resynth_x
      self._tmp_audio = audio.TempAudio(x, self.audio_sr)
      self._tmp_audio.play()
    return play_cbk_

  def export_cbk(self, *args, **kwargs):  # pylint: disable=W0613
    """Export callback"""
    try:
      j = self.sample_object.sdt_params_()
    except AttributeError:
      messagebox.showerror(
        "Not analyzed",
        "You have to analyse an audio file before "
        "exporting the parameters"
      )
      return
    logging.debug("SDT JSON: %s", j)
    filename = filedialog.asksaveasfilename(
      title="Save JSON file",
      initialdir=self.filedialog_dir_save,
      initialfile=os.path.basename(
        os.path.splitext(self.filedialog_file)[0]
      ),
      defaultextension=".json",
      filetypes=[
        ("JSON", ".json"),
        ("Text", ".txt"),
      ],
    )
    if filename:
      logging.info("Saving JSON: %s", filename)
      self.filedialog_dir_save = os.path.dirname(filename)
      try:
        with open(filename, "w") as f:
          json.dump(j, f, indent=2)
      except Exception as e:  # pylint: disable=W0703
        messagebox.showerror(type(e).__name__, str(e))
      else:
        messagebox.showinfo(
          "Saved", "Saved JSON to file:\n{}".format(filename)
        )
