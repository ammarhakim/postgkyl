"""Module including FLASH reader class"""

import math
import numpy as np
import tables
from typing import Tuple

# FLASH variable names
# dens : the density in g/cc
# tele : the electron temperature in K
# tion : same but for the ions
# velx : the fluid velocity in x direction
# vely : the fluid velocity in y direction
# temp : the overall fluid temperature in K
# pres : the pressure in dyn/cm^2
# ye
# sumy
#
# The last two variables are used to retrieve the ion and electron
# density in /cc:
# n_ele = ye * Na * dens
# n_ion = sumy * Na * dens
# where Na=6.02e23 is the Avogadro number.
#
# The average ionisation Z' and average atomic mass A' can be found by:
# Z' = ye/sumy
# A' = 1/sumy


class FlashH5Reader(object):
  """Provides a framework to read FLASH h5 output"""

  def __init__(self, file_name: str, var_name: str, ctx: dict = None, **kwargs) -> None:
    self._file_name = file_name
    self.var_name = var_name

    self.ctx = ctx

  def is_compatible(self) -> bool:
    out = False
    try:
      fh = tables.open_file(self._file_name, "r")
    except:
      return False
    # end
    if "coordinates" in fh.root:
      out = True
    # end
    fh.close()
    return out

  def _read_frame(self) -> tuple:
    fh = tables.open_file(self._file_name, "r")
    coord = fh.root["coordinates"].read().transpose()
    bsize = fh.root["block size"].read().transpose()
    ntype = fh.root["node type"].read().transpose()
    bdata = fh.root[self.var_name].read().transpose()

    nxb, nyb, _, N = bdata.shape
    res = bsize.min(axis=1)
    lower = (coord - bsize / 2).min(axis=1)
    upper = (coord + bsize / 2).max(axis=1)

    nxax = math.floor((upper[0] - lower[0]) / (res[0] / nxb))
    nyax = math.floor((upper[1] - lower[1]) / (res[1] / nyb))
    data = np.zeros((nxax, nyax))
    for b in range(N):
      if ntype[b] == 1:
        mult = np.ceil(bsize[:, b] / res)
        idxx = math.floor((coord[0, b] - bsize[0, b] / 2 - lower[0]) / res[0] * nxb)
        idxy = math.floor((coord[1, b] - bsize[1, b] / 2 - lower[1]) / res[1] * nyb)
        for i in range(nxb):
          for j in range(nyb):
            data[
                idxx + i * int(mult[0]) : idxx + (i + 1) * int(mult[0]) + 1,
                idxy + j * int(mult[1]) : idxy + (j + 1) * int(mult[1]) + 1,
            ] = bdata[i, j, 0, b]
          # end
        # end
      # end
    # end
    fh.close()
    return data.shape, lower[:2], upper[:2], data[..., np.newaxis]

  # ---- Exposed functions ----
  def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
    cells, lower, upper, data = self._read_frame()
    num_dims = len(cells)
    grid = [np.linspace(lower[d], upper[d], cells[d] + 1) for d in range(num_dims)]

    return grid, data
