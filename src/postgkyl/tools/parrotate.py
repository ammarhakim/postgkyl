
from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
  from postgkyl import GData
#end


def parrotate(data: GData, rotator: GData, rotate_coords: str = "0:3",
      overwrite=False, stack=False) -> Tuple[list, np.ndarray]:
  """Function to rotate input array into coordinate system parallel to rotator array
  For two arrays u and v, where v is the rotator, operation is (u dot v_hat) v_hat.

  Parameters:
  data -- input GData object being rotated
  rotator -- GData object used for the rotation
  rotate_coords -- optional input to specify a different set of coordinates in the rotator array used
  for the rotation (e.g., if rotating to the local magnetic field of a finite volume simulation, rotate_coords='3:6')

  Notes:
  Assumes three component fields, and that the number of components is the last dimension.
  For a three-component field, the output is a new vector
  whose components are (u_{v_x}, u_{v_y}, u_{v_z}), i.e.,
  the x, y, and z components of the vector u parallel to v.
  """
  if stack:
    overwrite = stack
    print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
  # end
  grid = data.get_grid()
  values = data.get_values()
  # Because rotate_coords is an input string, need to split and parse it to get the right coordinates
  s = rotate_coords.split(":")
  valuesrot = rotator.get_values()[..., slice(int(s[0]), int(s[1]))]

  outrot = np.zeros(values.shape)
  # Assumes three component fields and that the number of components is the last dimension
  try:
    outrot[..., 0] = np.sum(values*valuesrot, axis=-1)/(np.sum(valuesrot*valuesrot, axis=-1))*valuesrot[..., 0]
    outrot[..., 1] = np.sum(values*valuesrot, axis=-1)/(np.sum(valuesrot*valuesrot, axis=-1))*valuesrot[..., 1]
    outrot[..., 2] = np.sum(values*valuesrot, axis=-1)/(np.sum(valuesrot*valuesrot, axis=-1))*valuesrot[..., 2]
  except IndexError:
    print(
        f"parrotate: rotation failed due to different numbers of components, data numComponets = '{values.shape[-1]:d}', rotator numComponents = '{rotator.shape[-1]:d}'"
    )
    quit()
  # end
  if overwrite:
    data.push(grid, outrot)

  return grid, outrot
