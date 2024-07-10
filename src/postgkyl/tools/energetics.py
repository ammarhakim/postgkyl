"""Postgkyl module for separating energy components."""

import numpy as np

from postgkyl.data import GData
from postgkyl.tools import get_p, get_ke, magsq


def energetics(
    data_elc: GData, data_ion: GData, data_field: GData, overwrite=False, stack=False
):
  """Function to separate components of the energy.

  Works for both species and EM fields and separates into constituent parts (species
  energy -> thermal + kinetic, field energy -> electric + magnetic)

  Args: XXX needs refactoring; see laguerre_compose.py
    data_elc: GData
      input GData object for electrons
    data_ion: GData
      input GData object for ions
    data_field: GData
      input GData object for EM fields

  Notes:
    Assumes two-species plasma
  """
  if stack:
    overwrite = stack
    print(
        "Deprecation warning: The 'stack' parameter",
        "is going to be replaced with 'overwrite'",
    )
  # end
  # Grid is the same for each of the input objects
  grid = data_field.get_grid()
  values_field = data_field.get_values()
  # Output array is a seven component field
  #   1) Electron thermal
  #   2) Electron kinetic
  #   3) Ion thermal
  #   4) Ion kinetic
  #   5) Electric
  #   6) Magnetic
  #   7) Total
  out = np.zeros(values_field[..., :7].shape)

  grid, pre = get_p(data_elc)
  grid, kee = get_ke(data_elc)
  grid, pri = get_p(data_ion)
  grid, kei = get_ke(data_ion)
  # Can compute magnitude squared of electric and magnetic fields with magsq diagnostic
  grid, esq = magsq(data_field, coords="0:3")
  grid, bsq = magsq(data_field, coords="3:6")

  out[..., 0] = np.squeeze(pre)
  out[..., 1] = np.squeeze(kee)
  out[..., 2] = np.squeeze(pri)
  out[..., 3] = np.squeeze(kei)
  out[..., 4] = np.squeeze(esq / 2.0)
  out[..., 5] = np.squeeze(bsq / 2.0)
  out[..., 6] = np.squeeze(pre + kee + pri + kei + esq / 2.0 + bsq / 2.0)
  return grid, out
