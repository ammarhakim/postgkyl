import numpy as np
from typing import Union

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.data import GData
from postgkyl.tools import input_parser
# ----------------------------------------------------------------------

def laguerre_compose(in_f : Union[GData, tuple],
                     in_mom : Union[GData, tuple],
                     out_f : GData = None) -> tuple:
  """
  Compose PKPM expansion coefficients into a single f
  """
  in_f_grid, in_f_values = input_parser(in_f)
  _, in_mom_values = input_parser(in_mom)

  x, vpar = in_f_grid[0], in_f_grid[1]
  vperp = np.copy(vpar)

  x_cc = (x[:-1]+x[1:])/2
  vpar_cc = (vpar[:-1]+vpar[1:])/2
  vperp_cc = (vpar[:-1]+vpar[1:])/2

  _, _, vperp_3D = np.meshgrid(x_cc, vpar_cc, vperp_cc, indexing='ij')

  rho = in_mom_values[..., 0]
  p_perp = in_mom_values[..., 2]
  T_m = p_perp/rho

  F0 = in_f_values[..., 0]
  G = in_f_values[..., 1]

  F1 = F0 - (G.transpose()/T_m).transpose()

  s = vperp_3D.shape
  f = np.zeros(s)

  for i in range(s[0]):
    for j in range(s[1]):
      for k in range(s[2]):
        f[i,j,k] += F0[i,j] / (2*np.pi*T_m[i]) \
          * np.exp(-vperp_3D[i, j, k]**2/2/T_m[i])
        f[i,j,k] += F1[i,j] / (2*np.pi*T_m[j]) \
          * np.exp(-vperp_3D[i, j, k]**2/2/T_m[i]) \
            * (1 - vperp_3D[i, j, k]**2/2/T_m[i])
      #end
    #end
  #end

  f = f[..., np.newaxis]
  if (out_f):
    out_f.push([x, vpar, vperp], f)
  #end
  return [x, vpar, vperp], f
#end