
from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

from postgkyl.tools.parrotate import parrotate
if TYPE_CHECKING:
  from postgkyl import GData
#end



def perprotate(data: GData, rotator: GData, rotate_coords: str = "0:3",
      overwrite=False, stack=False) -> Tuple[list, np.ndarray]:
  """Function to rotate input array into coordinate system perpendicular to rotator array
  For two arrays u and v, where v is the rotator, operation is u - (u dot v_hat) v_hat.
  Uses the diagnostic parrotate.py to compute (u dot v_hat) v_hat.

  Parameters:
  data -- input GData object being rotated
  rotator -- GData object used for the rotation
  rotate_coords -- optional input to specify a different set of coordinates in the rotator array used
  for the rotation (e.g., if rotating to the local magnetic field of a finite volume simulation, rotate_coords='3:6')

  Notes:
  Assumes three component fields, and that the number of components is the last dimension.
  """
  if stack:
    overwrite = stack
    print(
        "Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'"
    )
  # end
  grid = data.get_grid()
  values = data.get_values()

  outrot = np.zeros_like(values)
  outrot = values - parrotate(data, rotator, rotate_coords)
  if overwrite:
    data.push(grid, outrot)
  #end

  return grid, outrot
