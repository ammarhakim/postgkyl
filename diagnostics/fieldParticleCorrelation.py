#!/usr/bin/env python
"""
Postgkyl module for field-particle correlations

Reference:
Klein & Howes, 2016, https://arxiv.org/abs/1607.01738v1
"""
import numpy
import exceptions

def Ce(f, g, N, mode='center'):
    """Calculate the field-particle correlation
    
    Parameters:
    f(x_0, v, t) -- modified distribution function
    g(x_0, t)    -- field
    N            -- number of time steps for averaging
    mode         -- averaging mode; 'forward',
                    'backward', and 'center' (default)

    Returns:
    Numpy array with dimension of time x velocity
    """
    length = g.shape[0]
    if mode == 'forward':
        offset = int(0)
    elif mode == 'backward':
        offset = int(N)
    elif mode == 'center':
        offset = int(numpy.floor(N/2))
    else:
        raise exceptions.RuntimeError(
            "fieldParticleC: Mode '{}' is not supported!\n  Supported modes are:\n    'forward' \n    'backward'\n    'center' (default)".format(mode))

    if len(g.shape) == 1:
        C = numpy.zeros(f.shape)
    else:
        C = numpy.zeros(f.shape[:-1])

    for i in numpy.arange(length-N)+offset:
        for j in range(N):
            if len(g.shape) == 1:
                C[i, ...] += f[i+j-offset, ...]*g[i+j-offset]
            elif g.shape[-1] == 2:
                C[i, ...] += f[i+j-offset, ..., 0]*g[i+j-offset, ..., 0] + \
                             f[i+j-offset, ..., 1]*g[i+j-offset, ..., 1]
            else:
                C[i, ...] += f[i+j-offset, ..., 0]*g[i+j-offset, ..., 0] + \
                             f[i+j-offset, ..., 1]*g[i+j-offset, ..., 1] + \
                             f[i+j-offset, ..., 2]*g[i+j-offset, ..., 2]


    return -C/N