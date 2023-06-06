import sys
import numpy as np

def _findNearestIndex(array, value):
  if array is None:
    raise TypeError("The index value is float but the 'array' from which to select the neares value is not specified.")
  #end
  idx = np.searchsorted(array, value)
  if idx == len(array):
    return int(idx-2)
  elif idx > 0:
    return int(idx-1)
  else:
    return int(idx)
  #end
#end

def _findCellIndex(array, value):
  if array is None:
    raise TypeError("The index value is float but the 'array' from which to select the neares value is not specified.")
  #end
  idx = np.searchsorted(array, value)
  return int(idx)
#end

def _stringToIndex(value, array=None, nodal=False):
  if isinstance(value, str):
    if value.isdigit():
      return int(value)
    else:
      if nodal:
        return _findCellIndex(array, float(value))
      else:
        return _findNearestIndex(array, float(value))
      #end
    #end
  else:
    raise TypeError('Value is not string')
  #end
#end

def idxParser(value, array=None, nodal=False):
  idx = None
  if isinstance(value, int):
    idx = value
  elif isinstance(value, float):
    if nodal:
      idx = _findCellIndex(array, value)
    else:
      idx = _findNearestIndex(array, value)
    #end
  else:
    if isinstance(value, str):
      if len(value.split(',')) > 1:
        idxs = value.split(',')
        idx = tuple([_stringToIndex(i, array, nodal) for i in idxs])
      elif len(value.split(':')) == 2:
        idxs = value.split(':')
        if idxs[0] == '':
          idxs[0] = str(0)
        #end
        if idxs[1] == '':
          idxs[1] = str(len(array))
        #end
        try:
          if int(idxs[1]) < 0:
            idxs[1] = str(len(array) + int(idxs[1]) + 1)
          #end
        except ValueError:
          pass
        idx = slice(_stringToIndex(idxs[0], array, nodal),
                    _stringToIndex(idxs[1], array, nodal))
      else:
        idx = _stringToIndex(value, array, nodal)
      #end
    #end
  #end

  return idx
#end   
