#!/usr/bin/env python
"""
Postgkyl module for rotating an array into the coordinate system perpendicular to another array
"""
import numpy as np
import postgkyl.diagnostics as diag

def perprotate(data, rotator, rotateCoords='0:3', overwrite=False):
    """Function to rotate input array into coordinate system perpendicular to rotator array
    For two arrays u and v, where v is the rotator, operation is u - (u dot v_hat) v_hat.
    Uses the diagnostic parrotate.py to compute (u dot v_hat) v_hat.

    Parameters:
    data -- input GData object being rotated
    rotator -- GData object used for the rotation
    rotateCoords -- optional input to specify a different set of coordinates in the rotator array used 
    for the rotation (e.g., if rotating to the local magnetic field of a finite volume simulation, rotateCoords='3:6')

    Notes:
    Assumes three component fields, and that the number of components is the last dimension.
    """
    grid = data.getGrid()
    values = data.getValues()

    outrot = np.zeros(values.shape)
    outrot = values - diag.parrotate(data, rotator, rotateCoords)
    if overwrite:
        data.push(grid, outrot)
    else:
        return grid, outrot
    #end
#end
