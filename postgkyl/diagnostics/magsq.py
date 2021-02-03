#!/usr/bin/env python
"""
Postgkyl module for computing the magnitude squared of an array
"""
import numpy as np

def magsq(data, overwrite=False):
    """Function to compute the magnitude squared of an array

    Parameters:
    data -- input array

    Notes:
    Assumes that the number of components is the last dimension.

    """
    grid = data.getGrid()
    values = data.getValues()
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
