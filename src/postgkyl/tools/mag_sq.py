from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

from postgkyl.utils import input_parser
if TYPE_CHECKING:
  from postgkyl import GData
# end


def mag_sq(dat: GData | Tuple[list, np.ndarray], coords: str = "0:3",
    output: GData | None = None) -> Tuple[list, np.ndarray]:
  """Function to compute the magnitude squared of an array

  Parameters:
    data
      input GData data structure
    coords
      specific coordinates to compute magnitude squared of by default assume a three
      component field and that you want the magnitude squared of the those three
      components

  Notes:
    Assumes that the number of components is the last dimension.

  """
  in_grid, in_values = input_parser(dat)

  # Because coords is an input string, need to split and parse it to get the right
  # coordinates.
  s = coords.split(":")
  values = in_values[..., slice(int(s[0]), int(s[1]))]
  # Output is a scalar, so dimensionality should not include number of components.
  out = np.zeros(values[..., 0].shape)
  out = np.sum(values*values, axis=-1)
  out = out[..., np.newaxis]

  if output:
    output.push(in_grid, out)
  # end
  return in_grid, out
