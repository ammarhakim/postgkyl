from __future__ import annotations

import numpy as np
from typing import Tuple,TYPE_CHECKING, Union

if TYPE_CHECKING:
  from postgkyl import GData


def input_parser(in_data: Union[GData, tuple]) -> Tuple[list, np.ndarray]:
  if isinstance(in_data, GData):
    return in_data.get_grid(), in_data.get_values()
  elif isinstance(in_data, tuple):
    return in_data[0], in_data[1]
  else:
    raise TypeError("Input must be either GData class or a tuple of numpy arrays.")
  # end