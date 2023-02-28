"""Test IPython utilities"""
import itertools
import unittest
from unittest import mock
from unittest.mock import patch

import sklearn.base
from chromatictools import unittestmixins
from IPython import display as ipd

from sample import ipython
from sample.evaluation import random


class TestIPython(unittestmixins.AssertDoesntRaiseMixin, unittest.TestCase):
  """Test IPython utilities"""

  def test_webaudio(self):
    """Test WebAudio widget"""
    with self.assert_doesnt_raise():
      x, fs, _ = random.BeatsGenerator(seed=1234).audio()
      ipython.WebAudio(x=x, rate=fs)

  def test_time_this(self):
    """Test ipython timer"""
    with self.assert_doesnt_raise():
      with ipython.time_this(title="<h1>Test</h1>"):
        pass

  def test_collapsible_model_params(self):
    """Test collapsible model params HTML widget"""

    class C(sklearn.base.BaseEstimator):
      """Dummy estimator container class for the test
      
      Args:
        c: Inner estimator
        d: Parameter"""

      class InnerC(sklearn.base.BaseEstimator):

        def __init__(self, a: float = 1, b: float = 0):
          self.a = a
          self.b = b
          super().__init__()

      def __init__(self, c=InnerC(), d: float = 0, **kwargs):
        self.c = c
        self.d = d
        super().__init__(**kwargs)

    with self.assert_doesnt_raise():
      ipython.CollapsibleModelParams(C())

  @patch("IPython.display.display")
  def test_play_foreach(self, mock_display: mock.Mock):
    """Test play widget"""
    it = itertools.product(
        (None, 0),
        (None, "0"),
        (None, [0]),
    )
    for i, k, y in it:
      audio = y is not None
      label = i is not None or k is not None
      kws = {"i": i, "k": k, "y": y}
      with self.subTest(**kws):
        mock_display.reset_mock()
        ipython.LabelAndPlayForeach(audio_kws={"rate": 8000})(**kws)
        # from IPython import embed; embed()
        with self.subTest(test="count"):
          self.assertEqual(mock_display.call_count, label or audio)
        if not (label or audio):
          continue
        with self.subTest(test="n_widgets"):
          self.assertEqual(len(mock_display.call_args_list[0][0]),
                           label + audio)
        if label:
          with self.subTest(test="label"):
            self.assertIsInstance(mock_display.call_args_list[0][0][0],
                                  ipd.HTML)
        if audio:
          with self.subTest(test="audio"):
            self.assertIsInstance(mock_display.call_args_list[0][0][-1],
                                  ipd.Audio)
