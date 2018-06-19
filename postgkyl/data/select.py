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
    lo, up = data.getBounds()
    lo = np.array(lo)  # copy the lower boundaries
    up = np.array(up)  # copy the upper boundaries
    values = data.getValues()
    numDims = data.getNumDims()
    idxValues = [slice(0, values.shape[d]) for d in range(numDims+1)]
    
    # Loop for coordinates
    for d, coord in enumerate(coords):
        if d < numDims and coord is not None:
            coord = idxParser(coord, grid[d])
            # dz = grid[d][1] - grid[d][0]
            if isinstance(coord, int):
                grid[d] = grid[d][coord, np.newaxis]
                idxValues[d] = coord
            elif isinstance(coord, slice):
                    grid[d] = grid[d][idx]
                    idxValues[d] = idx
            # Adjust the grid span
            lo[d] = grid[d].min() - 0.5*dz
            up[d] = grid[d].max() + 0.5*dz

    # Select components
    if comp is not None:
        idxValues[-1] = idxParser(comp)

    valuesOut = values[idxValues]
    # Adding a dummy dimension indicies
    for d, coord in enumerate(coords):
        if d < len(grid) and coord is not None and len(grid[d]) == 1:
            valuesOut = np.expand_dims(valuesOut, d)
    # Ddding a dummy component index
    if len(grid) == len(valuesOut.shape):
        valuesOut = valuesOut[..., np.newaxis]

    if stack:
        data.pushGrid(grid, lo, up)
        data.pushValues(valuesOut)
    else:
        return grid, valuesOut
                
