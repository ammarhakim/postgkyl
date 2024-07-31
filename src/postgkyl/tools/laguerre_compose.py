"""Postgkyl module for combining the two laguerre components F0 and F1.

Within Gkeyll, this is mostly use for working with the PKPM data.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

from postgkyl.utils import input_parser
if TYPE_CHECKING:
  from postgkyl import GData
# end


def laguerre_compose(in_f: GData | Tuple[list, np.ndarray],
    in_T_m: GData | Tuple[list, np.ndarray],
    out_f: GData | None = None) -> Tuple[list, np.ndarray]:
  """Compose PKPM expansion coefficients into a single f.

  Compose the full distribution function f(x, v_par, v_perp) out of the
  Laguerre expansion coefficients F0(x, v_par), F1(x, v_par) and the
  PKPM moments to calculate the T(x) over m.

  Jimmy Juno's slides: https://drive.google.com/file/d/1548tLF9o7vyW3bkrsq6FvAMV-8XJvKtY/view

  Args:
    in_f: GData or NumPy array
      2-component Laguerre expansion coefficients.
    in_T_m: GData or NumPy array
      PKPM T over m.
    out_f: GData = None
      (Optional) GData to store output.

  Returns:
    A tuple of grid (which is itself a tuple of nupy arrays for each dimension) and a
    NumPy array with values.
  """
  in_f_grid, in_f_values = input_parser(in_f)
  _, in_T_m_values = input_parser(in_T_m)

  x, vpar = in_f_grid[0], in_f_grid[1]
  vperp = np.copy(vpar)

  x_cc = (x[:-1] + x[1:])/2
  vpar_cc = (vpar[:-1] + vpar[1:])/2
  vperp_cc = (vpar[:-1] + vpar[1:])/2

  _, _, vperp_3D = np.meshgrid(x_cc, vpar_cc, vperp_cc, indexing="ij")

  F0 = in_f_values[..., 0]
  G = in_f_values[..., 1]
  T_m = in_T_m_values[..., 0]

  F1 = F0 - (G.transpose()/T_m).transpose()

  # Ading the np.newaxis allows the subsequent np.multiply (called when
  # doing * on numpy arrays) to work. The arrays need to have the same
  # number of axis, e.g., one can not multiply (3, 3) and (3,) arrays
  # but can multiply (3, 3) with (3, 1) or (1, 3).
  F0, F1 = F0[..., np.newaxis], F1[..., np.newaxis]
  T_m = T_m[..., np.newaxis, np.newaxis]

  # Hardcoded for l=0, n=0,1 in
  # https://drive.google.com/file/d/1548tLF9o7vyW3bkrsq6FvAMV-8XJvKtY/view
  f = (F0 + F1*(1 - vperp_3D**2/2/T_m))/(2*np.pi*T_m) * np.exp(-(vperp_3D**2)/2/T_m)

  f = f[..., np.newaxis]  # Adding the component index

  if out_f:
    out_f.push([x, vpar, vperp], f)
  # end
  return [x, vpar, vperp], f
