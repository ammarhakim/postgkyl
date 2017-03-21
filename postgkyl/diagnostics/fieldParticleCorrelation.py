#!/usr/bin/env python
"""
Postgkyl module for field-particle correlations

Reference:
Klein & Howes, 2016, https://arxiv.org/abs/1607.01738v1
"""
import numpy

def Ce(N, f, E, v, dv=None, q=1, mode='center'):
    """Calculate the field-particle correlation
    
    Parameters:
    N -- number of time steps for averaging
    f(x0, v, t) -- distribution function
    E(x0, t) -- electric field
    v -- velocity field
    dv -- velocity field differences (default: 1 in each direction)
    q -- electric charge (default: 1)
    mode -- averaging mode; 'forward',
           'backward', and 'center' (default)

    Returns:
    Numpy array with dimension of time x velocity
     q v^2 \partial_v f E

    Raises:
    NameError -- when 'mode' is specified
    RuntimeError -- when input parameters don't have correct dimensions

    Notes:
    Averaging 
    """
    if mode == 'forward':
        offset = int(0)
    elif mode == 'backward':
        offset = int(N)
    elif mode == 'center':
        offset = int(numpy.floor(N/2))
    else:
        raise NameError(
            "Ce: Mode '{}' is not supported!\n  Supported modes are:\n    'forward' \n    'backward'\n    'center' (default)".format(mode))

    length = f.shape[0]
    vDim = len(f.shape)-1 # 1st dimension is time
    if vDim > 3:
        raise RuntimeError(
            "Ce: velocity dimension appears to be {}.\nNote that Ce is expecting distribution function only in velocity space for fixed 'x'".format(vDim))

    C = numpy.zeros(f.shape)

    if dv is None:
        dv = numpy.full(vDim, 1)

    # calculate the gradient 
    # (should be replaced by a propper DG derivation...)
    axis = tuple(numpy.arange(vDim) + 1)
    if vDim == 1:
        df = numpy.array(numpy.gradient(f, dv, axis=axis, edge_order=2))
    else:
        df = numpy.array(numpy.gradient(f, *dv, axis=axis, edge_order=2))

    # get v^2
    if vDim == 1:
        v = numpy.squeeze(v)
        v2 = v**2
    elif vDim == 2:
        VX, VY = numpy.meshgrid(v[0, :], v[1, :], indexing='ij')
        v2 = VX**2 + VY**2
    else:
        VX, VY, VZ = numpy.meshgrid(v[0, :], v[1, :], v[2, :], indexing='ij')
        v2 = VX**2 + VY**2 + VZ**2

    for i in numpy.arange(length - N) + offset:
        for j in range(N):
            if vDim == 1:
                f = numpy.squeeze(f)
                E = numpy.squeeze(E)
                C[i, ...] += 0.5*q*v2*df[i+j-offset, ...]*E[i+j-offset]
            elif vDim == 2:
                C[i, ...] += 0.5*q*v2*(df[0, i+j-offset, ...]*E[0, i+j-offset]+ 
                                       df[1, i+j-offset, ...]*E[1, i+j-offset])
            else:
                C[i, ...] += 0.5*q*v2*(df[0, i+j-offset, ...]*E[0, i+j-offset]+ 
                                       df[1, i+j-offset, ...]*E[1, i+j-offset]+ 
                                       df[2, i+j-offset, ...]*E[2, i+j-offset])

    C = numpy.array(-C/N)

    # masking
    mask = numpy.zeros(f.shape)
    if N > 1:
        if mode == 'forward':
            mask[-N+1 :, ...] = 1
        elif mode == 'backward':
            mask[: N-1, ...] = 1
        elif mode == 'center':
            mask[: int(numpy.floor(N/2))-1, ...] = 1
            mask[-int(numpy.ceil(N/2))-1 :, ...] = 1

    Cm = numpy.ma.masked_array(C, mask)

    return Cm
