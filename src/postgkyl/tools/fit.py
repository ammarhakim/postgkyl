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


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
  """A * exp(-0.5 * ((x - mu) / sigma)²)"""
  return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def power(x: np.ndarray, a: float, n: float, b: float) -> np.ndarray:
  """a * x^n + b"""
  return a * x**n + b


def sinusoid(x: np.ndarray, A: float, omega: float, phi: float, C: float) -> np.ndarray:
  """A * sin(omega * x + phi) + C"""
  return A * np.sin(omega * x + phi) + C


def tanh_transition(x: np.ndarray, A: float, x0: float, w: float, C: float) -> np.ndarray:
  """A * tanh((x - x0) / w) + C"""
  return A * np.tanh((x - x0) / w) + C


RPN_OPERATORS: frozenset = frozenset({'+', '-', '*', '/', '**', '^'})

RPN_FUNCTIONS: dict[str, Callable] = {
    'exp':   np.exp,
    'log':   np.log,
    'ln':    np.log,
    'log10': np.log10,
    'sin':   np.sin,
    'cos':   np.cos,
    'tan':   np.tan,
    'sqrt':  np.sqrt,
    'abs':   np.abs,
    'tanh':  np.tanh,
}

_SPATIAL_VARS: frozenset = frozenset({'x', 'y', 'z'})


def rpn_param_names(expression: str) -> list[str]:
  """Return the free parameter names from an RPN expression, in order of first appearance."""
  names = []
  for tok in expression.split():
    if tok in _SPATIAL_VARS or tok in RPN_OPERATORS or tok in RPN_FUNCTIONS:
      continue
    try:
      float(tok)
    except ValueError:
      if tok not in names:
        names.append(tok)
  return names


def rpn_ndim(expression: str) -> int:
  """Return 1 or 2 depending on whether 'y' appears as a spatial variable."""
  return 2 if 'y' in expression.split() else 1


def _rpn_make_func(expression: str) -> Callable:
  """Build a curve_fit-compatible callable from an RPN expression string."""
  tokens = expression.split()
  param_names = rpn_param_names(expression)
  ndim = rpn_ndim(expression)

  def _func(xdata, *param_values):
    ns: dict = dict(zip(param_names, param_values))
    if ndim == 1:
      ns['x'] = np.asarray(xdata, dtype=float)
    else:
      ns['x'] = np.asarray(xdata[0], dtype=float)
      ns['y'] = np.asarray(xdata[1], dtype=float)

    stack = []
    for tok in tokens:
      if tok in RPN_OPERATORS:
        b, a = stack.pop(), stack.pop()
        if   tok == '+':        stack.append(a + b)
        elif tok == '-':        stack.append(a - b)
        elif tok == '*':        stack.append(a * b)
        elif tok == '/':        stack.append(a / b)
        else:                   stack.append(a ** b)  # ** or ^
      elif tok in RPN_FUNCTIONS:
        stack.append(RPN_FUNCTIONS[tok](stack.pop()))
      elif tok in ns:
        stack.append(ns[tok])
      else:
        stack.append(float(tok))

    result = stack[0]
    ref = ns.get('x', ns.get('y'))
    if np.ndim(result) == 0 and ref is not None:
      result = np.full_like(ref, float(result))
    return np.asarray(result, dtype=float)

  return _func


FIT_FUNCTIONS: dict[str, Callable] = {
    "linear": linear,
    "quadratic": quadratic,
    "plane": plane,
    "quadratic2d": quadratic2d,
    "exp_plateau": exp_plateau,
    "gaussian": gaussian,
    "power": power,
    "sinusoid": sinusoid,
    "tanh_transition": tanh_transition,
}

# Number of spatial dimensions each fit type operates on
FIT_NDIM: dict[str, int] = {
    "linear": 1,
    "quadratic": 1,
    "plane": 2,
    "quadratic2d": 2,
    "exp_plateau": 1,
    "gaussian": 1,
    "power": 1,
    "sinusoid": 1,
    "tanh_transition": 1,
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
  if fit_type in FIT_FUNCTIONS:
    func = FIT_FUNCTIONS[fit_type]
    n_params = func.__code__.co_argcount - 1
  else:
    toks = set(fit_type.split())
    if not (toks & (RPN_OPERATORS | set(RPN_FUNCTIONS))):
      raise ValueError(f"fit_type '{fit_type}' not recognized. Choose from: {list(FIT_FUNCTIONS)}")
    func = _rpn_make_func(fit_type)
    n_params = len(rpn_param_names(fit_type))

  if p0 is None:
    p0 = np.ones(n_params)

  params, cov = opt.curve_fit(func, xdata, ydata, p0=p0)

  residual = ydata - func(xdata, *params)
  ss_res = np.sum(residual**2)
  ss_tot = np.sum((ydata - np.mean(ydata))**2)
  R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

  return params, cov, R2
