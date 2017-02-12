#!/usr/bin/env python
"""
Postgkyl module with diagnostics functions
"""

import numpy

def partFieldC(f, g, N):
    length = g.shape[0]

    C = numpy.zeros((length-N, f.shape[1]))
    for i in range(length-N):
        for j in range(N):
            C[i, :] = C[i, :] + f[i+j, :]*g[i+j]

    return -C/N
