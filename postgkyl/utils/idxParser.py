import sys
import numpy as np

def _findNearestIndex(value, array=None):
    if array is None:
        raise TypeError("The index value is float but the 'array' from which to select the neares value is not specified.")
    idx = np.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) <
                    np.fabs(value - array[idx])):
        return int(idx-1)
    else:
        return int(idx)

def _stringToIndex(value, array=None):
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        else:
            return _findNearestIndex(float(value), array)
    else:
        raise TypeError('Value is not string')

def idxParser(value, array=None):
    if isinstance(value, int):
        idx = value
    if isinstance(value, float):
        idx = _findNearestIndex(value, array)

    if sys.version_info[0] < 3 and isinstance(value, unicode):
      value = str(value)
    if isinstance(value, str):
        if len(value.split(',')) > 1:
            idxs = value.split(',')
            idx = tuple([_stringToIndex(i, array) for i in idxs])
        elif len(value.split(':')) == 2:
            idxs = value.split(':')
            idx = slice(_stringToIndex(idxs[0], array),
                        _stringToIndex(idxs[1], array))
        else:
            idx = _stringToIndex(value, array)

    return idx
     
