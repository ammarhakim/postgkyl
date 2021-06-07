#!/usr/bin/env python
"""
Postgkyl module for computing relative change in a quantity
"""
import numpy as np

def rel_change(dataset0, dataset, comp=None, overwrite=False, stack=False):
    """Function to compute the relative change in a dataset compared to another
    dataset, i.e. (dataset - dataset0)/dataset0

    Notes:
    Assumes user wishes to perform this operation component-wise.
    Also assumes the reference division should be performed with respect to a single
    component (i.e., for energetics, divide by the total energy, 
    not an individual component of the energy)
    """
    # Grid is the same for each of the input objects
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = dataset.getGrid()
    values = dataset.getValues()
    values0 = dataset0.getValues()
    out = np.zeros(values.shape)
    for i in range(0, out.shape[-1]):
        if comp is not None:
            out[..., i] = (values[..., i] - values0[..., i])/values0[..., int(comp)]
        else:
            out[..., i] = (values[..., i] - values0[..., i])/values0[..., i]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end
