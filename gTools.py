#!/usr/bin/env python
"""
Postgkyl module with random useful stuff :)
"""

import numpy

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
