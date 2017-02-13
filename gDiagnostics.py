#!/usr/bin/env python
"""
Postgkyl module with diagnostics functions
"""

import numpy
import exceptions

def fieldParticleC(f, g, N, mode='forward'):
    """Calculate the field-particle correlation
    
    Inputs:
    f(x_0, v, t) -- modified distribution function
    g(x_0, t)    -- field
    N            -- number of time steps for averaging
    mode         -- averaging mode; 'forward' (default),
                    'backward', and 'center'

    Returns:
    Numpy array with dimension of time x velocity

    Reference:
    Klein & Howes, 2016, https://arxiv.org/abs/1607.01738v1
    """
    length = g.shape[0]
    if mode == 'forward':
        offset = 0
    elif mode == 'backward':
        offset = N
    elif mode == 'center':
        ofset = numpy.floor(N/2)
    else:
        raise exceptions.RuntimeError(
            "fieldParticleC: Mode '{}' is not supported!\n  Supported modes are:\n    'forward' (default)\n    'backward'\n    'center'.".format(mode))

    C = numpy.zeros((length, f.shape[1]))
    for i in numpy.arange(length-N)+offset:
        for j in range(N):
            C[i, :] = C[i, :] + f[i+j-offset, :]*g[i+j-offset]

    return -C/N
