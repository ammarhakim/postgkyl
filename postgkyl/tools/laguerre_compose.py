import numpy as np
from typing import Union

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.data import GData
from postgkyl.tools import _input_parser
# ----------------------------------------------------------------------

def laguerre_compose(in_f : Union[GData, tuple],
                     in_mom : Union[GData, tuple],
                     out_f : GData = None) -> tuple:
  """
  Compose PKPM expansion coefficients into a single f
  """
  in_f_grid, in_f_values = _input_parser(in_f)
  _, in_mom_values = _input_parser(in_mom)

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

  F0, F1 = F0[..., np.newaxis], F1[..., np.newaxis]
  T_m = T_m[..., np.newaxis, np.newaxis]

  f =  ( F0 + F1 * (1 - vperp_3D**2/2/T_m) ) \
    / (2*np.pi*T_m) * np.exp(-vperp_3D**2/2/T_m)

  f = f[..., np.newaxis]
  if (out_f):
    out_f.push([x, vpar, vperp], f)
  #end
  return [x, vpar, vperp], f
#end