"""Postgkyl module for accumulating current."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

from postgkyl.utils import input_parser

if TYPE_CHECKING:
  from postgkyl import GData
#end


def accumulate_current(data: GData | Tuple[list, np.ndarray], qbym: bool = False,
    overwrite=False, stack=False) -> Tuple[list, np.ndarray]:
  """Computes the current from an arbitrary number of input species.

  Args:
    data: GData or grid and values
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
  grid, values = input_parser(data)
  out = np.zeros_like(values)
  factor = 0.0
  if qbym and data.mass and data.charge is not None:
    factor = data.charge/data.mass
  else:
    factor = -1.0
  # end
  out = factor*values
  if overwrite:
    data.push(grid, out)
  return grid, out
