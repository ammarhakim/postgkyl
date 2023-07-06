import numpy as np
from typing import Union

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.data import GData
from postgkyl.tools import input_parser
# ----------------------------------------------------------------------

def transform_frame(in_f : Union[GData, tuple],
                    in_u : Union[GData, tuple],
                    c_dim : int,
                    out_f : GData = None) -> tuple:
  """
  Shift distribution function to a different frame of reference
  """
  in_f_grid, in_f_values = input_parser(in_f)
  _, in_u_values = input_parser(in_u)
  v_dim = len(in_f_grid) - c_dim
  out_grid = np.meshgrid(*in_f_grid, indexing='ij')

  if c_dim == 1:
    print(in_f_grid[0].shape[0])
    for i in range(in_f_grid[0].shape[0]-1):
      for v_idx in range(v_dim):
        out_grid[c_dim+v_idx][i, ...] += in_u_values[i, v_idx]
      #end
    #end
  elif c_dim == 2:
    for i in range(in_f_grid[0].shape[0]):
      for j in range(in_f_grid[0].shape[1]):
        for v_idx in range(v_dim):
          out_grid[c_dim+v_idx][i, j, ...] += in_u_values[i, j, v_idx]
        #end
      #end
    #end
  else:
    for i in range(in_f_grid[0].shape[0]):
      for j in range(in_f_grid[0].shape[1]):
        for k in range(in_f_grid[0].shape[2]):
          for v_idx in range(v_dim):
            out_grid[c_dim+v_idx][i, j, k, ...] += \
              in_u_values[i, j, k, v_idx]
          #end
        #end
      #end
    #end
  #end

  if (out_f):
    out_f.push(out_grid, in_f_values)
  #end
  return out_grid, in_f_values
#end