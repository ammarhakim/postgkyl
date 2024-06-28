#!/usr/bin/env python
"""
Postgkyl module for computing the magnitude squared of an array
"""
import numpy as np


def mag_sq(in_data=None, in_grid=None, in_values=None, coords="0:3", overwrite=False):
  """Function to compute the magnitude squared of an array

  Parameters:
  data -- input GData data structure
  coords -- specific coordinates to compute magnitude squared of
            by default assume a three component field and that you
            want the magnitude squared of the those three components

  Notes:
  Assumes that the number of components is the last dimension.

  """
  if in_data:
    in_grid = in_data.get_grid()
    in_values = in_data.get_values()
  # end
  # Because coords is an input string, need to split and parse it to get the right coordinates
  s = coords.split(":")
  values = in_values[..., slice(int(s[0]), int(s[1]))]
  # Output is a scalar, so dimensionality should not include number of components.
  out = np.zeros(values[..., 0].shape)
  out = np.sum(values * values, axis=-1)
  out = out[..., np.newaxis]
  if overwrite:
    in_data.push(in_grid, out)
  else:
    return in_grid, out
  # end


# end
