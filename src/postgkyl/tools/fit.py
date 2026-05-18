"""Postgkyl module for curve fitting using scipy."""

import numpy as np
import scipy.optimize as opt
from typing import Callable, Tuple


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
  return a * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  return a * x**2 + b * x + c


def plane(XY: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
  x, y = XY
  return a*x + b*y + c


def quadratic2d(XY: np.ndarray, a: float, b: float, c: float,
    d: float, e: float, f: float) -> np.ndarray:
  """a*x² + b*y² + c*x*y + d*x + e*y + f"""
  x, y = XY
  return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


def exp_plateau(x: np.ndarray, A: float, b: float, C: float) -> np.ndarray:
  """A*exp(b*x) + C  (plateaus at C as b*x → -∞, or at A+C as b*x → +∞)"""
  return A * np.exp(b * x) + C


FIT_FUNCTIONS: dict[str, Callable] = {
    "linear": linear,
    "quadratic": quadratic,
    "plane": plane,
    "quadratic2d": quadratic2d,
    "exp_plateau": exp_plateau,
}

# Number of spatial dimensions each fit type operates on
FIT_NDIM: dict[str, int] = {
    "linear": 1,
    "quadratic": 1,
    "plane": 2,
    "quadratic2d": 2,
    "exp_plateau": 1,
}


def fit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fit_type: str = "linear",
    p0: list | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
  """Fit data using scipy curve_fit with the specified model.

  Parameters
  ----------
  xdata : ndarray
      For 1D fits: shape (N,). For 2D fits: shape (2, N) where rows are the
      two independent variables flattened.
  ydata : ndarray
      Dependent variable, shape (N,).
  fit_type : str
      One of the keys in FIT_FUNCTIONS.
  p0 : list, optional
      Initial guess for the fit parameters.

  Returns
  -------
  params : ndarray
  cov : ndarray
  R2 : float
  """
  if fit_type not in FIT_FUNCTIONS:
    raise ValueError(f"fit_type '{fit_type}' not recognized. Choose from: {list(FIT_FUNCTIONS)}")

  func = FIT_FUNCTIONS[fit_type]
  n_params = func.__code__.co_argcount - 1
  if p0 is None:
    p0 = np.ones(n_params)

  params, cov = opt.curve_fit(func, xdata, ydata, p0=p0)

  residual = ydata - func(xdata, *params)
  ss_res = np.sum(residual**2)
  ss_tot = np.sum((ydata - np.mean(ydata))**2)
  R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

  return params, cov, R2
