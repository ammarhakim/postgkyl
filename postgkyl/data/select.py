import numpy as np

# Find the neares value to the input in a specified narray
def _findNearest(array, value):
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

# Find the neares value to the input in a specified narray and return
# its index
def _findNearestIdx(array, value):
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# Helper function to convert strings to integers and find the neares
# index for the float input
def _getIdx(idx, grid):
    if isinstance(idx, str):
        if idx.isdigit():
            idx = int(idx)
        else:
            idx = float(idx)
    if isinstance(idx, int):
        return idx
    elif isinstance(idx, float):
        return int(_findNearestIdx(grid, idx))
    elif isinstance(idx, str):
        if idx.isdigit():
            return int(idx)
        else:
            return int(_findNearestIdx(grid, float(idx)))
    else:
        raise TypeError("'idx' is neither int, float or, str")

def select(gdata, comp=None,
           coord0=None, coord1=None, coord2=None,
           coord3=None, coord4=None, coord5=None):
    """Selects parts of the GData.

    Allows to select only a part of GData (both coordinates and
    components).  Allows for numpy slices, selecting multiple
    components, and using both indicies (integer) and values (float).

    Atributes:
        gdata (GData)
        coord0-5 (index, value, or slice (e.g. '1:5')
        comp (index, slice (e.g. '1:5'), or multiple (e.g. '1,5')
    """
    coords = (coord0, coord1, coord2, coord3, coord4, coord5)
    grid = gdata.peakGrid()
    grid = list(grid)  # copy the grid
    lo, up = gdata.getBounds()
    lo = np.array(lo)  # copy the lower boundaries
    up = np.array(up)  # copy the upper boundaries
    values = gdata.peakValues()
    numDims = gdata.getNumDims()
    idxValues = [slice(0, values.shape[d]) for d in range(numDims+1)]
    
    # Loop for coordinates
    for d, coord in enumerate(coords):
        if d < numDims and coord is not None:
            dz = grid[d][1] - grid[d][0]
            if isinstance(coord, int) or isinstance(coord, float):
                idx = _getIdx(coord, grid[d])
                grid[d] = grid[d][idx, np.newaxis]
                idxValues[d] = idx
            elif isinstance(coord, str):
                if len(coord.split(':')) == 2:
                    idxs = coord.split(':')
                    idx = slice(_getIdx(idxs[0], grid[d]),
                                _getIdx(idxs[1], grid[d]))
                    grid[d] = grid[d][idx]
                    idxValues[d] = idx
                else:
                    idx = _getIdx(coord, grid[d])
                    grid[d] = grid[d][idx, np.newaxis]
                    idxValues[d] = idx
            # Adjust the grid span
            lo[d] = grid[d].min() - 0.5*dz
            up[d] = grid[d].max() + 0.5*dz
    # Select components
    if comp is not None:
        if isinstance(comp, int) or isinstance(comp, float):
            idx = _getIdx(comp, grid[d])
            idxValues[-1] = idx
        elif isinstance(comp, str):
            if len(comp.split(',')) > 1:
                idxs = comp.split(',')
                idx = tuple([int(i) for i in idxs])
                idxValues[-1] = idx
            elif len(comp.split(':')) == 2:
                idxs = comp.split(':')
                idx = slice(int(idxs[0]), int(idxs[1]))
                idxValues[-1] = idx
            else:
                idx = int(comp)
                idxValues[-1] = idx
            
    gdata.pushGrid(grid, lo, up)

    valuesOut = values[idxValues]
    # Adding a dummy dimension indicies
    for d, coord in enumerate(coords):
        if coord is not None and len(grid[d]) == 1:
            valuesOut = np.expand_dims(valuesOut, d)
    # Ddding a dummy component index
    if len(grid) == len(valuesOut.shape):
        valuesOut = valuesOut[..., np.newaxis]
    gdata.pushValues(valuesOut)
                
