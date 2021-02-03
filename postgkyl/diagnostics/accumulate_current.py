#!/usr/bin/env python
"""
Postgkyl module for accumulating current
"""
import numpy as np

def accumulate_current(data, qbym=False, overwrite=False):
    """Function to compute current from an arbitrary number of input species

    Parameters:
    data -- input field
            NOTE: These are GData objects which include metadata such as charge and mass
    qbym -- optional input for multiplying by charge/mass ratio instead of just charge
            NOTE: Should be true for fluid data

    """
    values = data.getValues()
    out = np.zeros(values.shape)
    factor = 0.0
    if (qbym and dat.mass is not None and dat.charge is not None):
        factor = dat.charge/dat.mass
    #elif (dat.charge is not None): 
    #    factor = dat.charge
    else:
        factor = -1.0
    #end
    out = factor*values
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end
