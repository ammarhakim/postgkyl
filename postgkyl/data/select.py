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

def select(data, comp=None,
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
    grid, lo, up = data.peakGrid()
    grid = list(grid)
    lo = np.array(lo)
    up = np.array(up)
    values = data.peakValues()
    numDims = data.getNumDims()
    idxValues = [slice(0, values.shape[d]) for d in range(numDims+1)]
    
    for d, coord in enumerate(coords):
        if d < numDims and coord is not None:
            dx = grid[d][1] - grid[d][0]
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
            lo[d] = grid[d].min() - 0.5*dx
            up[d] = grid[d].max() + 0.5*dx
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
            
    data.pushGrid(grid, lo, up)
    valuesOut = values[idxValues]
    for d, nc in enumerate(data.getNumCells()):
        if nc == 1:
            valuesOut = np.expand_dims(valuesOut, d)
    if len(valuesOut.shape) == numDims:
        valuesOut = valuesOut[..., np.newaxis]
    data.pushValues(valuesOut)
                
