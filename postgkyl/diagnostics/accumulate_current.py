#!/usr/bin/env python
"""
Postgkyl module for accumulating current
"""
import numpy as np

def accumulate_current(data, qbym=False, stack=False):
    """Function to compute current from an arbitrary number of input species

    Parameters:
    *args -- input arguments 
             NOTE: These should be GData objects which include parameters such as charge and mass
    qbym -- optional input for multiplying by charge/mass ratio instead of just charge
            NOTE: Should be true for fluid data

    """

    values = data.getValues()

    factor = 0.0

    if (qbym and dat.mass is not None and dat.charge is not None):
        factor = dat.charge/dat.mass
    #elif (dat.charge is not None): 
    #    factor = dat.charge
    else:
        factor = -1.0
    #end
    out = factor*values

    return out
#end