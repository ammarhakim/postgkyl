from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from postgkyl.utils import input_parser

if TYPE_CHECKING:
  from postgkyl import GData


def load_plot_data(data: GData | Tuple[list, np.ndarray]) -> tuple[list, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
  """Load grid/values and derive dimensional metadata used by plot backends."""
  grid_in, values = input_parser(data)
  grid = grid_in.copy()

  if isinstance(data, tuple):
    if len(grid) == len(values.shape):
      num_dims = len(values.squeeze().shape)
    else:
      num_dims = len(values[..., 0].squeeze().shape)
    # end
    lg = len(grid)
    lower, upper, cells = np.zeros(lg), np.zeros(lg), np.zeros(lg)
    for d in range(lg):
      lower[d] = np.min(grid[d])
      upper[d] = np.max(grid[d])
      if len(grid[d].shape) == 1:
        cells[d] = len(grid[d])
      else:
        cells[d] = len(grid[d][d])
      # end
    # end
  else:  # GData
    num_dims = data.get_num_dims(squeeze=True)
    lower, upper = data.get_bounds()
    cells = data.get_num_cells()
  # end

  return grid, values, num_dims, np.asarray(lower), np.asarray(upper), np.asarray(cells)
