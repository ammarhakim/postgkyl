"""Postgkyl module for curve fitting using scipy."""

import numpy as np
import scipy.optimize as opt
from typing import Callable, Tuple


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
  return a * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  return a * x**2 + b * x + c


FIT_FUNCTIONS: dict[str, Callable] = {
    "linear": linear,
    "quadratic": quadratic,
}


def fit(
    x: np.ndarray,
    y: np.ndarray,
    fit_type: str = "linear",
    p0: list | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
  """Fit data using scipy curve_fit with the specified model.

  Parameters
  ----------
  x : array-like
      Independent variable.
  y : array-like
      Dependent variable.
  fit_type : str
      One of the keys in FIT_FUNCTIONS ('linear', 'quadratic').
  p0 : list, optional
      Initial guess for the fit parameters.

  Returns
  -------
  params : ndarray
      Optimal fit parameters.
  cov : ndarray
      Estimated covariance of params.
  R2 : float
      Coefficient of determination.
  """
  if fit_type not in FIT_FUNCTIONS:
    raise ValueError(f"fit_type '{fit_type}' not recognized. Choose from: {list(FIT_FUNCTIONS)}")

  func = FIT_FUNCTIONS[fit_type]
  n_params = func.__code__.co_argcount - 1  # subtract x argument
  if p0 is None:
    p0 = np.ones(n_params)

  params, cov = opt.curve_fit(func, x, y, p0=p0)

  residual = y - func(x, *params)
  ss_res = np.sum(residual**2)
  ss_tot = np.sum((y - np.mean(y))**2)
  R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

  return params, cov, R2
