#!/usr/bin/env python
"""
Postgkyl sub-module for custom field manipulation
"""
import numpy
import math


def rotationMatrix(vector):
  """Calculate rotation matrix

  Inputs:
  vector

  Returns:
  3x3 rotation matrix (numpy array)
  """
  rot = numpy.zeros((3, 3))
  norm = numpy.abs(vector)
  k = vector / norm  # direction unit vector

  # normalization
  norm2 = numpy.sqrt(k[1] * k[1] + k[2] * k[2])
  norm3 = numpy.sqrt(
      (k[1] * k[1] + k[2] * k[2]) ** 2
      + k[0] * k[0] * k[1] * k[1]
      + k[0] * k[0] * k[2] * k[2]
  )

  rot[0, :] = k
  rot[1, 0] = 0
  rot[1, 1] = -k[2] / norm2
  rot[1, 2] = k[1] / norm2
  rot[2, 0] = (k[1] * k[1] + k[2] * k[2]) / norm3
  rot[2, 1] = -k[0] * k[1] / norm3
  rot[2, 2] = -k[0] * k[2] / norm3

  return rot


def findNearest(array, value):
  idx = numpy.searchsorted(array, value)
  if idx > 0 and (
      idx == len(array)
      or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
  ):
    return array[idx - 1]
  else:
    return array[idx]


def findNearestIdx(array, value):
  idx = numpy.searchsorted(array, value)
  if idx > 0 and (
      idx == len(array)
      or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
  ):
    return idx - 1
  else:
    return idx


def fixGridSlice(
    grid,
    values,
    mode="idx",
    fix1=None,
    fix2=None,
    fix3=None,
    fix4=None,
    fix5=None,
    fix6=None,
):
  """Fix specified coordinates and decrease dimensionality

  Parameters:
  grid -- array of coordinates
  values -- array of field values
  mode -- changes input between direct index ('idx', default) and
          coordinate value ('value')
  fix1 -- fixes the first coordinate to provided index (default None)
  fix2 -- fixes the second coordinate to provided index (default None)
  fix3 -- fixes the third coordinate to provided index (default None)
  fix4 -- fixes the fourth coordinate to provided index (default None)
  fix5 -- fixes the fifth coordinate to provided index (default None)
  fix6 -- fixes the sixth coordinate to provided index (default None)

  Returns:
  gridOut -- coordinates with decreased number of dimensions
  valuesOut -- field values with decreased number of dimensions

  Example:
  By fixing an x-index (fix1), 1X1V simulation data transforms
  to 1D velocity profile.

  Note:
  Fixing higher dimensions than available in the data has no effect.
  """
  fix = (fix1, fix2, fix3, fix4, fix5, fix6)

  num_dims = len(values.shape) - 1
  idxGrid = []
  idxValues = [slice(0, values.shape[d]) for d in range(num_dims)]

  for i, idx in enumerate(fix):
    if i < num_dims:
      if idx is not None:
        if mode == "value":
          idx = findNearestIdx(grid[i], float(idx))
        idxValues[i] = int(idx)
      else:
        idxGrid.append(int(i))
  return grid[idxGrid], values[idxValues]
