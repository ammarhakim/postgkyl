import numpy as np

from postgkyl.utils import idxParser


def select(data, comp=None, stack=False,
           coord0=None, coord1=None, coord2=None,
           coord3=None, coord4=None, coord5=None):
    """Selects parts of the GData.

    Allows to select only a part of GData (both coordinates and
    components).  Allows for numpy slices, selecting multiple
    components, and using both indicies (integer) and values (float).

    Atributes:
        data (GData)
        coord0-5 (index, value, or slice (e.g. '1:5')
        comp (index, slice (e.g. '1:5'), or multiple (e.g. '1,5')
    """
    coords = (coord0, coord1, coord2, coord3, coord4, coord5)
    grid = data.getGrid()
    grid = list(grid)  # copy the grid
    values = data.getValues()
    numDims = data.getNumDims()
    idxValues = [slice(0, values.shape[d]) for d in range(numDims+1)]
    
    # Loop for coordinates
    for d, coord in enumerate(coords):
        if d < numDims and coord is not None:
            if values.shape[d] == len(grid[d]):
                nodal = False
            else:
                nodal = True
            idx = idxParser(coord, grid[d], nodal)
            if isinstance(idx, int):
                if nodal:
                    grid[d] = grid[d][slice(idx, idx+1)]
                else:
                    grid[d] = grid[d][idx, np.newaxis]
            elif isinstance(idx, slice):
                if nodal:
                    grid[d] = grid[d][slice(idx.start, idx.stop+1)]
                else:
                    grid[d] = grid[d][idx]
            else:
                raise TypeError("The coordinate select can be only single index (int) or a slice")
            idxValues[d] = idx
            

    # Select components
    if comp is not None:
        idxValues[-1] = idxParser(comp)

    valuesOut = values[idxValues]
    # Adding a dummy dimension indicies
    for d, coord in enumerate(coords):
        if d < numDims and coord is not None and len(grid[d]) == 1:
            valuesOut = np.expand_dims(valuesOut, d)
    # Ddding a dummy component index
    if numDims == len(valuesOut.shape):
        valuesOut = valuesOut[..., np.newaxis]

    if stack:
        data.pushGrid(grid)
        data.pushValues(valuesOut)
    else:
        return grid, valuesOut
                
