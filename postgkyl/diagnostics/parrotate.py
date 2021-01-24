#!/usr/bin/env python
"""
Postgkyl module for rotating an array into the coordinate system parallel to another array
"""
import numpy as np

def parrotate(data, rotator, stack=False):
    """Function to rotate input array into coordinate system parallel to rotator array
    For two arrays u and v, where v is the rotator, operation is (u dot v_hat) v_hat.

    Parameters:
    data -- input array
    rotator -- array being used for the rotation

    Notes:
    Assumes three component fields, and that the number of components is the last dimension.
    For a three-component field, the output is a new vector
    whose components are (u_{v_x}, u_{v_y}, u_{v_z}), i.e.,
    the x, y, and z components of the vector u parallel to v. 
    """
    outrot = np.zeros(data.shape)
    # Assumes three component fields and that the number of components is the last dimension
    try:
        outrot[...,0] = np.tensordot(data,rotator,axes=-1)/(np.tensordot(rotator, rotator,axes=-1))*rotator[...,0]
        outrot[...,1] = np.tensordot(data,rotator,axes=-1)/(np.tensordot(rotator, rotator,axes=-1))*rotator[...,1]
        outrot[...,2] = np.tensordot(data,rotator,axes=-1)/(np.tensordot(rotator, rotator,axes=-1))*rotator[...,2]
    except IndexError:
        print("parrotate: rotation failed due to different numbers of components, data numComponets = '{:d}', rotator numComponents = '{:d}'".format(data.shape[-1], rotator.shape[-1]))
        quit()
    #end
    return outrot
#end