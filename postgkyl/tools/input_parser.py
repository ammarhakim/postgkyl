from postgkyl.data import GData
from typing import Union

def _input_parser(in_data: Union[GData, tuple]) -> tuple:
  if type(in_data) == GData:
    return in_data.getGrid(), in_data.getValues()
  elif type(in_data) == tuple:
    return in_data[0], in_data[1]
  else:
    raise TypeError('Input must be either GData class or a tuple of numpy arrays.')
  #end
#end
