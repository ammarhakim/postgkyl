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
    k = vector / norm # direction unit vector

    # normalization
    norm2 = numpy.sqrt(k[1]*k[1] + k[2]*k[2])
    norm3 = numpy.sqrt((k[1]*k[1] + k[2]*k[2])**2 + 
                       k[0]*k[0]*k[1]*k[1] + 
                       k[0]*k[0]*k[2]*k[2])

    rot[0, :] =  k
    rot[1, 0] =  0
    rot[1, 1] = -k[2]/norm2
    rot[1, 2] =  k[1]/norm2
    rot[2, 0] = (k[1]*k[1] + k[2]*k[2])/norm3
    rot[2, 1] = -k[0]*k[1]/norm3
    rot[2, 2] = -k[0]*k[2]/norm3

    return rot

def findNearest(array, value):
    idx = numpy.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) 
                    < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
def findNearestIdx(array, value):
    idx = numpy.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) 
                    < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def fixCoordSlice(coords, values,
                  fix1=None, fix2=None, fix3=None,
                  fix4=None, fix5=None, fix6=None):
    """Fix specified coordinates and decrease dimensionality

    Parameters:
    coords -- array of coordinates
    values -- array of field values
    fix1 -- fixes the first coordinate to provided index (default None)
    fix2 -- fixes the second coordinate to provided index (default None)
    fix3 -- fixes the third coordinate to provided index (default None)
    fix4 -- fixes the fourth coordinate to provided index (default None)
    fix5 -- fixes the fifth coordinate to provided index (default None)
    fix6 -- fixes the sixth coordinate to provided index (default None)

    Returns:
    coordsOut -- coordinates with decreased number of dimensions
    valuesOut -- field values with decreased number of dimensions

    Example:
    By fixing an x-index (fix1), 1X1V simulation data transforms
    to 1D velocity profile.

    Note:
    Fixing higher dimensions than available in the data has no effect.
    """
    fix = (fix1, fix2, fix3, fix4, fix5, fix6)
    coordsOut = numpy.copy(coords)
    valuesOut = numpy.copy(values)
    for i, val in reversed(list(enumerate(fix))):
        if val is not None and len(values.shape) > i:
            # turn N-D coords into correct 1D coord array
            temp = coords[i]
            coords1D = numpy.linspace(temp.min(), temp.max(),
                                      temp.shape[i])
            idx = findNearestIdx(coords1D, float(val))
            # create for mask compressing
            mask = numpy.zeros(values.shape[i])
            mask[int(idx)] = 1
            # delete coordinate matrices for the fixed coordinate
            coordsOut = numpy.delete(coordsOut, i, 0)
            coordsOut = numpy.compress(mask, coordsOut, axis=i+1)  
            coordsOut = numpy.squeeze(coordsOut)

            valuesOut = numpy.compress(mask, valuesOut, axis=i) 
            valuesOut = numpy.squeeze(valuesOut)
    return coordsOut, valuesOut
