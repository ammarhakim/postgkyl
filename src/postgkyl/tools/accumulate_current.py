"""Postgkyl module for accumulating current."""

from typing import Tuple
import numpy as np

def accumulate_current(data, qbym: bool = False,
    overwrite=False, stack=False) -> Tuple[list, np.ndarray]:
  """Computes the current from an arbitrary number of input species.

  Args:
    data: postgkyl.GData
      input field
      NOTE: These are GData objects which include metadata such as charge and mass
    qbym: bool = False
      optional input for multiplying by charge/mass ratio instead of just charge
      NOTE: Should be true for fluid data

    XXX overwrite and stack need refactoring; see laguerre_compose.py
  """
  if stack:
    overwrite = stack
    print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
  # end
  grid = data.grid
  values = data.values
  out = np.zeros(values.shape)
  factor = 0.0
  if qbym and data.mass and data.charge is not None:
    factor = data.charge / data.mass
  else:
    factor = -1.0
  # end
  out = factor * values
  if overwrite:
    data.push(grid, out)
  return grid, out
  # end
