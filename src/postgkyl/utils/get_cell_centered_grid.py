

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
  from postgkyl import GData
# end

def get_cell_centered_grid(grid : list, cells: np.ndarray):
  """Return cell-centered grid from nodal grid.
  
  Args:
    grid: list of NumPy arrays representing the grid coordinates
    cells: NumPy array representing the number of cells in each dimension

  Returns:
    list of NumPy arrays representing the cell-centered grid coordinates

  Example:
  grid_in, values = input_parser(GDataObject)
  grid_out = get_cell_centered_grid(grid_in, values.shape)
  """
  
  num_dims = len(grid)
  grid_out = []
  if num_dims != len(cells):  # sanity check
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  # end
  for d in range(num_dims):
    if len(grid[d].shape) == 1:
      if grid[d].shape[0] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[0] == cells[d] + 1:
        grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
      else:
        raise ValueError("Something is terribly wrong...")
      # end
    else:
      if grid[d].shape[d] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[d] == cells[d] + 1:
        if num_dims == 1:
          grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
        else:
          grid_out.append(0.5 * (grid[d][:-1, :-1] + grid[d][1:, 1:]))
        # end
      else:
        raise ValueError("Something is terribly wrong...")
      # end
    # end
  # end
  return grid_out