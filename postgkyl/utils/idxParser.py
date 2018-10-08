import sys
import numpy as np

def _findNearestIndex(array, value):
    if array is None:
        raise TypeError("The index value is float but the 'array' from which to select the neares value is not specified.")
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return int(idx-1)
    else:
        return int(idx)

def _findCellIndex(array, value):
    if array is None:
        raise TypeError("The index value is float but the 'array' from which to select the neares value is not specified.")
    idx = np.searchsorted(array, value)
    return int(idx-1)

def _stringToIndex(value, array=None, nodal=True):
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        else:
            if nodal:
                return _findCellIndex(array, float(value))
            else:
                return _findNearestIndex(array, float(value))
    else:
        raise TypeError('Value is not string')

def idxParser(value, array=None, nodal=True):
    idx = None
    if isinstance(value, int):
        idx = value
    elif isinstance(value, float):
        if nodal:
            idx = _findCellIndex(array, value)
        else:
            idx = _findNearestIndex(array, value)
    else:
        if sys.version_info[0] < 3 and isinstance(value, unicode):
            value = str(value)
        if isinstance(value, str):
            if len(value.split(',')) > 1:
                idxs = value.split(',')
                idx = tuple([_stringToIndex(i, array, nodal) for i in idxs])
            elif len(value.split(':')) == 2:
                idxs = value.split(':')
                idx = slice(_stringToIndex(idxs[0], array, nodal),
                            _stringToIndex(idxs[1], array, nodal))
            else:
                idx = _stringToIndex(value, array, nodal)

    return idx
     
