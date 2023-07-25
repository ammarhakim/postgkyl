import numpy as np
import os.path

class Read_gkyl_adios(object):
  """Provides a framework to read gkyl Adios output
  """

  def __init__(self, file_name : str, **kwargs) -> None:
    self.file_name = file_name

  #end

  def _is_compatible(self) -> bool:
    if True:
      return True
    #end
    return False
  #end