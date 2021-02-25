"""Tests related to SMS"""
import unittest
import numpy as np
from scipy import special
from sample import regression
from tests import utils


def noisy_hinge(
  x: np.ndarray,
  a: float,
  k: float,
  q: float,
  n: float,
  seed: int = 42,
) -> np.ndarray:
  """Synthesize a noisy hinge function sample

  Args:
    x (array): Independent variable
    a (float): Knee point
    k (float): Slope
    q (float): Intercept
    n (float): Noise magnitude (multiplier)
    seed (int): Gaussian noise seed

  Returns:
    array: Noisy hinge sample :py:data:`h(x) + N(0, h(x)*n)`"""
  np.random.seed(seed)
  y = special.exp10(regression.hinge_function(x, a, k, q) / 20)
  return 20 * np.log10(y + np.random.randn(*y.shape) * y * n)


class TestRegression(utils.SignificantPlacesAssertMixin, unittest.TestCase):
  """Tests related to regression"""
  def setUp(self) -> None:
    """Initialize test sample and model"""
    self.a = 37
    self.k = -np.pi
    self.q = -20
    self.n = .25
    self.x = (np.arange(1024) * 2001 / 44100).reshape((-1, 1))
    self.y = noisy_hinge(self.x, self.a, self.k, self.q, self.n)
    self.hr = regression.HingeRegression()

  def test_approximately_correct(self):
    """Test that fitted parameters are almost equal to ground truth"""
    self.hr.fit(self.x, self.y)
    with self.subTest(variable="a"):
      self.assert_almost_equal_significant(self.hr.a_, self.a, places=1)
    with self.subTest(variable="k"):
      self.assert_almost_equal_significant(self.hr.k_, self.k, places=1)
    with self.subTest(variable="q"):
      self.assert_almost_equal_significant(self.hr.q_, self.q, places=1)

  @unittest.expectedFailure
  def test_linear_model_correct(self):
    """Test that linearly fitted parameters are almost equal to ground truth"""
    self.hr.linear_regressor.fit(self.x, self.y)
    with self.subTest(variable="k"):
      self.assert_almost_equal_significant(
        np.squeeze(self.hr.linear_regressor.coef_),
        self.k,
        places=1
      )
    with self.subTest(variable="q"):
      self.assert_almost_equal_significant(
        np.squeeze(self.hr.linear_regressor.intercept_),
        self.q,
        places=1
      )


if __name__ == "__main__":
  unittest.main()
