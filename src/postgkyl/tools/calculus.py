"""Postgkyl module for calculating integrals and derivatives."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
  from postgkyl import GData
# end


def integrate(data: GData, axis: int | tuple | str,
    overwrite=False, stack=False) -> Tuple[list, np.ndarray]:
  """Integrates Gkeyll data.

  Currently simply uses the NumPy dot function. True, DG integration should be
  implemented at some point.

  Args:
    data: GData
    axis: int, tuple or str
      Specify axis to integrate over

    XXX overwrite and stack need refactoring; see laguerre_compose.py
  """
  if stack:
    overwrite = stack
    print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
  # end
  grid = list(data.grid)
  values = np.copy(data.values)

  # Convert Python input to an input Numpy understands
  if axis is not None:
    if isinstance(axis, int):
      axis = tuple([axis])
    elif isinstance(axis, tuple):
      pass
    elif isinstance(axis, str):
      if len(axis.split(",")) > 1:
        axes = axis.split(",")
        axis = tuple([int(a) for a in axes])
      elif len(axis.split(":")) == 2:
        bounds = axis.split(":")
        # axis = np.zeros(bounds[1]-bounds[0], np.int)
        # axis += int(bounds[0])
        axis = tuple(range(bounds[0], bounds[1]))
      else:
        axis = tuple([int(axis)])
      # end
    else:
      raise TypeError(
          "'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')"
      )
    # end
  else:
    num_dims = data.get_num_dims()
    axis = tuple(range(num_dims))
  # end

  # Get dz elements
  dz = []
  for d, coord in enumerate(grid):
    dz.append(coord[1:] - coord[:-1])
    if len(coord) > 1 and len(coord) == values.shape[d]:
      dz[-1] = np.append(dz[-1], dz[-1][-1])
    # end
  # end

  # Integration assuming values are cell centered averages
  # Should work for nonuniform meshes
  for ax in sorted(axis, reverse=True):
    if len(grid[ax]) > 1:
      values = np.moveaxis(values, ax, -1)
      values = np.dot(values, dz[ax])
    else:
      values = values.mean(axis=ax)
    # end
  # end

  for ax in sorted(axis):
    grid[ax] = np.array([grid[ax].mean()])
    values = np.expand_dims(values, ax)
  # end

  if overwrite:
    data.push(grid, values)

  return grid, values
  # end


def grad():
  ...


def div():
  ...


def curl():
  ...
