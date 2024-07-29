"""Postgkyl module for fitting growth rates."""

import numpy as np
import scipy.optimize as opt
import sys
from typing import Callable, Tuple


def exp2(x: float, a: float, b: float) -> float:
  """Define custom exponential a*exp(2b*x)

  Args:
    x: float
      independent variable
    a: float
      scaling parameter
    b: float
      growth rate

  Notes:
    Energy (quantity^2) is often used for the growth-rate study,
    therefore the factor 2
  """
  return a*np.exp(2*b*x)


def fit_growth(x: np.ndarray, y: np.ndarray, function: Callable = exp2,
  min_N: int | None = None, p0: tuple = (1, 1)) -> Tuple[tuple, float, int]:
  """Fit function to continuously increasing region of data

  Parameters:
    x: NumPy array
      independet variable
    y: NumPy array
      dependent variable
    min_N: int
      minimal number of fitted points
    function: callable = exp2
      function to fit
    p0: tuple = (1, 1)
      initial guess

  Notes:
    The best is determined based on the coeficient of determination,
    R^2 https://en.wikipedia.org/wiki/Coefficient_of_determination
  """
  best_R2 = 0.0
  if min_N is None:
    min_N = int(len(x)/10)
  max_N = len(x)
  best_N = min_N
  best_params = p0

  max_x = x[-1]

  print(f"fit_growth: fitting region {min_N:d} -> {max_N:d}")
  for n in np.linspace(min_N, max_N - 1, max_N - min_N):
    n = int(n)
    xn = x[0:n]/max_x  # continuously increasing fitting region
    yn = y[0:n]
    try:
      params, _ = opt.curve_fit(function, xn, yn, best_params)
      residual = yn - function(xn, *params)
      ss_res = np.sum(residual**2)
      ss_tot = np.sum((yn - np.mean(yn))**2)
      R2 = 1 - ss_res/ss_tot
      if R2 > best_R2:
        best_R2 = R2
        best_params = params
        best_N = n
      # end
      percent = float(n - min_N) / (max_N - min_N)*100
      progress = "[" + int(percent / 10) * "=" + (10 - int(percent / 10)) * " " + "]"
      sys.stdout.write(
          f"\rgamma = {best_params[1] / max_x:+.5e} (current {params[1] / max_x:+.3e} R^2={R2:.3e})   {percent:6.2f}% done {progress}")
      sys.stdout.flush()
    except RuntimeError:
      print(f"fit_growth: curve_fit failed for N = {n:d}")
    # end
  # end
  best_params[1] = best_params[1]/max_x
  print(f"\ngamma = {best_params[1]:+.5e}")
  return best_params, best_R2, best_N
