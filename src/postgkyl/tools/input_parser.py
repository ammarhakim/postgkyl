from typing import Union

from postgkyl.data import GData


def input_parser(in_data: Union[GData, tuple]) -> tuple:
  if type(in_data) == GData:
    return in_data.get_grid(), in_data.get_values()
  elif type(in_data) == tuple:
    return in_data[0], in_data[1]
  else:
    raise TypeError("Input must be either GData class or a tuple of numpy arrays.")
  # end
