import numpy as np
from typing import Union

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.data import GData
from postgkyl.tools import _input_parser
# ----------------------------------------------------------------------


def laguerre_compose(
    in_f: Union[GData, tuple], in_T_m: Union[GData, tuple], out_f: GData = None
) -> tuple:
  """Compose PKPM expansion coefficients into a single f.

  Compose the full distribution function f(x, v_par, v_perp) out of the
  Laguerre expansion coefficients F0(x, v_par), F1(x, v_par) and the
  PKPM moments to calculate the T(x) over m.

  Args:
    in_f: 2-component Laguerre expansion coefficients.
    in_T_m: PKPM T over m.
    out_f: (Optional) GData to store output.

  Returns:
    A tuple of grid (which is itself a tuple of nupy arrays for each
    dimension) and a numpy array with values.
  """
  in_f_grid, in_f_values = _input_parser(in_f)
  _, in_T_m_values = _input_parser(in_T_m)

  x, vpar = in_f_grid[0], in_f_grid[1]
  vperp = np.copy(vpar)

  x_cc = (x[:-1] + x[1:]) / 2
  vpar_cc = (vpar[:-1] + vpar[1:]) / 2
  vperp_cc = (vpar[:-1] + vpar[1:]) / 2

  _, _, vperp_3D = np.meshgrid(x_cc, vpar_cc, vperp_cc, indexing="ij")

  F0 = in_f_values[..., 0]
  G = in_f_values[..., 1]
  T_m = in_T_m_values[..., 0]

  F1 = F0 - (G.transpose() / T_m).transpose()

  # Ading the np.newaxis allows the subsequent np.multiply (called when
  # doing * on numpy arrays) to work. The arrays need to have the same
  # number of axis, e.g., one can not multiply (3, 3) and (3,) arrays
  # but can multiply (3, 3) with (3, 1) or (1, 3).
  F0, F1 = F0[..., np.newaxis], F1[..., np.newaxis]
  T_m = T_m[..., np.newaxis, np.newaxis]

  # Hardcoded for l=0, n=0,1 in
  # https://drive.google.com/file/d/1548tLF9o7vyW3bkrsq6FvAMV-8XJvKtY/view?usp=sharing
  f = (
      (F0 + F1 * (1 - vperp_3D**2 / 2 / T_m))
      / (2 * np.pi * T_m)
      * np.exp(-(vperp_3D**2) / 2 / T_m)
  )

  f = f[..., np.newaxis]  # Adding the component index

  if out_f:
    out_f.push([x, vpar, vperp], f)
  # end
  return [x, vpar, vperp], f


# end
