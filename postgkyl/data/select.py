import numpy as np

def _findNearest(array, value):
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def _findNearestIdx(array, value):
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

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

def select(data, axis0=None, axis1=None, axis2=None,
           axis3=None, axis4=None, axis5=None, comp=None):
    """Selects parts of the GData.

    Allows to select only a part of GData (both coordinates and
    components).  Allows for numpy slices, selecting multiple
    components, and using both indicies (integer) and values (float).

    Atributes:
        data (GData)
        axis0-5 (index, value, slice (e.g. '1:5'), or multiple (e.g. '1,5')
    """
    axes = (axis0, axis1, axis2, axis3, axis4, axis5)
    grid, lo, up = data.peakGrid()
    grid = list(grid)
    lo = list(lo)
    up = list(up)
    values = data.peakValues()
    numDims = data.getNumDims()
    idxValues = [slice(0, values.shape[d]) for d in range(numDims+1)]
    
    for d, idx in enumerate(axes):
        if d < numDims and idx is not None:
            if isinstance(idx, int) or isinstance(idx, float):
                idx = _getIdx(idx, grid[d])
                grid[d] = grid[d][idx, np.newaxis]
                idxValues[d] = idx
            elif isinstance(idx, str):
                if len(idx.split(',')) > 1:
                    idxs = idx.split(',')
                    idx = tuple([_getIdx(i, grid[d]) for i in idxs])
                    grid[d] = grid[d][idx]
                    idxValues[d] = idx
                elif len(idx.split(':')) == 2:
                    idxs = component.split(':')
                    idx = slice(_getIdx(idx[0], grid[d]),
                                _getIdx(idx[0], grid[d]))
                    grid[d] = grid[d][idx]
                    idxValues[d] = idx                           
            lo[d] = grid[d].min()
            up[d] = grid[d].max()
        if d == numDims and idx is not None:
            if isinstance(idx, int) or isinstance(idx, float):
                idx = _getIdx(idx, grid[d])
                idxValues[d] = idx
            elif isinstance(idx, str):
                if len(idx.split(',')) > 1:
                    idxs = idx.split(',')
                    idx = tuple([_getIdx(i, grid[d]) for i in idxs])
                    idxValues[d] = idx
                elif len(idx.split(':')) == 2:
                    idxs = component.split(':')
                    idx = slice(_getIdx(idx[0], grid[d]),
                                _getIdx(idx[0], grid[d]))
                    idxValues[d] = idx  
            
    data.pushGrid(grid, lo, up)
    valuesOut = values[idxValues]
    for d, nc in enumerate(data.getNumCells()):
        if nc == 1:
            valuesOut = np.expand_dims(valuesOut, d)
    data.pushValues(valuesOut)
                
