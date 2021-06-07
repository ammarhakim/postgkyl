#!/usr/bin/env python
"""
Postgkyl module for computing the magnitude squared of an array
"""
import numpy as np

def magsq(data, coords='0:3', overwrite=False, stack=False):
    """Function to compute the magnitude squared of an array

    Parameters:
    data -- input GData data structure
    coords -- specific coordinates to compute magnitude squared of
              by default assume a three component field and that you 
              want the magnitude squared of the those three components

    Notes:
    Assumes that the number of components is the last dimension.

    """
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    # Because coords is an input string, need to split and parse it to get the right coordinates
    s = coords.split(':')
    values = data.getValues()[...,slice(int(s[0]), int(s[1]))]
    # Output is a scalar, so dimensionality should not include number of components.
    out = np.zeros(values[...,0].shape)
    out = np.sum(values*values, axis=-1)
    out = out[..., np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end
