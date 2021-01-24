#!/usr/bin/env python
"""
Postgkyl module for rotating an array into the coordinate system perpendicular to another array
"""
import numpy as np
import postgkyl.diagnostics.parrotate

def perprotate(data, rotator, stack=False):
    """Function to rotate input array into coordinate system perpendicular to rotator array
    For two arrays u and v, where v is the rotator, operation is u - (u dot v_hat) v_hat.
    Uses the diagnostic parrotate.py to compute (u dot v_hat) v_hat.

    Parameters:
    data -- input array
    rotator -- array being used for the rotation

    Notes:
    Assumes three component fields, and that the number of components is the last dimension.
    """
    grid = data.getGrid()
    values = data.getValues()
    valuesrot = rotator.getValues()

    outrot = np.zeros(values.shape)
    outrot = values - postgkyl.diagnostics.parrotate(values, valuesrot)

    return outrot
#end