"""Utilities for audio"""
import io
import threading

import numpy as np
import pygame
from pygame import mixer
from scipy.io import wavfile


class TempAudio:
  """Temporary audio from numpy array

  Args:
    x (array): Audio samples buffer
    sr (int): Sample rate
    format_ (str): Audio file format. Default is :data:`"wav"`"""
  lock = threading.Lock()

  def __init__(
      self,
      x: np.ndarray,
      sr: int,
  ):
    self.buf = io.BytesIO()
    self.x = x
    self.sr = sr
    wavfile.write(self.buf, sr, x)
    self.buf.seek(0)

  @staticmethod
  def close_pygame():
    try:
      mixer.music.unload()
    except pygame.error:
      pass
    mixer.quit()
    pygame.quit()

  def open_pygame(self):
    pygame.init()
    mixer.init(
        frequency=self.sr,
        channels=1,
    )

  def play(self):
    """Play the temporary audio with :mod:`pygame`"""
    with self.lock:
      if not self.buf.closed:
        self.close_pygame()
        self.open_pygame()
        mixer.music.load(self.buf)
        mixer.music.play()

  def __del__(self):
    """On delete, close buffer if not already closed"""
    with self.lock:
      if not self.buf.closed:
        self.buf.close()
