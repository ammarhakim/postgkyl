from postgkyl.data import GData
from typing import Union

def input_swith(in_data: Union[GData, tuple]) -> tuple:
  if type(in_data) == GData:
    return in_data.getGrid(), in_data.getValues()
  elif type(in_data) == tuple:
    return in_data
  else:
    raise TypeError('Input must be either GData class or a tuple of numpy arrays.')
  #end
#end
