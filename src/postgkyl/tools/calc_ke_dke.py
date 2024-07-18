"""Postgkyl module for calculating total kinetic energy."""

from typing import Tuple
import numpy as np

import postgkyl


def calc_ke_dke(root_file_name: str, init_frame: int, final_frame: int, dim: int,
    vol: float, init_time: float, final_time: float) -> Tuple[np.ndarray, np.ndarray]:
  """CalculateS all the total kinetic energy and the rate of dissipation of KE

  Args:
    root_file_name: str
      the name of the file before the numbers start
    init_frame: int
      is the first frame
    final_frame: int
      is the final frame
    dim: int
      gives the dimension of the simulation (2 = 2D, 3 = 3D)
    vol: float
      the volume of the grid

  Returns:
    kinetic energy and dissipation of KE
  """

  # calculate integrated kinetic energy
  ke = np.zeros((1, (final_frame - init_frame + 1)))
  dEk = ke
  f = postgkyl.GData(f"{root_file_name}{str(init_frame)}.bp")
  grid = f.get_grid()
  dx = grid[0][1] - grid[0][0]
  dy = grid[1][1] - grid[1][0]
  dt = (final_time - init_time + 1) / (final_frame - init_frame + 1)
  r = 0

  if dim == 3:
    dz = grid[2][1] - grid[2][0]
  else: # dim == 2:
    dz = 1

  for c in range(init_frame, final_frame + 1):
    frame = postgkyl.GData(f"root_file_name{c:d}.bp")
    data = frame.get_values()
    rho = data[..., 0]
    px = data[..., 1]
    py = data[..., 2]
    pz = data[..., 3]

    u = px / rho
    v = py / rho
    w = pz / rho

    e = rho * (u**2 + v**2 + w**2)
    ke[0, r] = np.sum(e, axis=(0, 1, 2))*dx*dy*dz*vol
    r += 1

  r = 0
  for i in range(init_frame, final_frame - 1):
    dEk[0, r] = -(ke[0, i + 1] - ke[0, i]) / dt
    r += 1
  # end

  return ke, dEk
