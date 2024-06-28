#!/usr/bin/env python
"""
Postgkyl module for accumulating current
"""
import numpy as np


def accumulate_current(data, qbym=False, overwrite=False, stack=False):
  """Function to compute current from an arbitrary number of input species

  Parameters:
  data -- input field
          NOTE: These are GData objects which include metadata such as charge and mass
  qbym -- optional input for multiplying by charge/mass ratio instead of just charge
          NOTE: Should be true for fluid data

  """
  if stack:
    overwrite = stack
    print(
        "Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'"
    )
  # end
  grid = data.get_grid()
  values = data.get_values()
  out = np.zeros(values.shape)
  factor = 0.0
  if qbym and data.mass is not None and data.charge is not None:
    factor = data.charge / data.mass
  # elif (dat.charge is not None):
  #    factor = dat.charge
  else:
    factor = -1.0
  # end
  out = factor * values
  if overwrite:
    data.push(grid, out)
  else:
    return grid, out
  # end
