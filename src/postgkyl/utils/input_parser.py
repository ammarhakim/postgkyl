"""Postgkyl module to unify inputs for various tools and diagnostics."""
from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
  from postgkyl import GData
# end
import postgkyl.data.gdata

def input_parser(data: GData | np.ndarray | Tuple[list, np.ndarray]) -> Tuple[list, np.ndarray]:
  """Utility function to parse input and return grid and values.

  Motivation for this funtion is to unify what input is used by Postgkyl tools and
  diagnostics. Sometimes it's beneficial to pass the internal GData class and in other
  situations it's more convenient to pass a grid and values.

  Args:
    data: GData | NumPy array | tuple of grid list and NumPy array
      Input ot be parsef

  Returns:
    grid: list of NumPy arrays
    values: NumPy array

  Raises:
    TypeError when wrong data type is provided
    ValueError dimensions of grid and values don't match
  """
  if isinstance(data, postgkyl.data.gdata.GData):
    return data.get_grid(), data.get_values()
  elif isinstance(data, np.ndarray):
    return (), data
  elif isinstance(data, tuple) or isinstance(data, list): # A little leeway
    if len(data) == 2:
      if not isinstance(data[0], list):
        raise TypeError("Input grid needs to be a list of NumPy arrays.")
      if not isinstance(data[1], np.ndarray):
        raise TypeError("Input values needs to be a NumPy array.")
      if len(data[0]) != len(data[1].shape) and len(data[0]) != len(data[1].shape)-1:
        raise ValueError("Input grid and valeus don't have the same number of dimesnions.")
      return data[0], data[1]
    else:
      raise TypeError("Input tuple needs to have two components: grid and values; {len(data):d} were provided.")
  else:
    raise TypeError("Input must be either GData class or a tuple of grid and values.")
  # end
