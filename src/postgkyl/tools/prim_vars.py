from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING

from postgkyl.utils import input_parser
if TYPE_CHECKING:
  from postgkyl import GData
# end


def get_density(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = in_values[..., 0, np.newaxis]

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_vx(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  out_values = in_values[..., 1, np.newaxis] / rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_vy(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  out_values = in_values[..., 2, np.newaxis] / rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_vz(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  out_values = in_values[..., 3, np.newaxis] / rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_vi(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  out_values = in_values[..., 1:4] / rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pxx(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vx = get_vx(in_mom)
  out_values = in_values[..., 4, np.newaxis] - rho*vx*vx

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pxy(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vx = get_vx(in_mom)
  _, vy = get_vy(in_mom)
  out_values = in_values[..., 5, np.newaxis] - rho*vx*vy

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pxz(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vx = get_vx(in_mom)
  _, vz = get_vz(in_mom)
  out_values = in_values[..., 6, np.newaxis] - rho*vx*vz

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pyy(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vy = get_vy(in_mom)
  out_values = in_values[..., 7, np.newaxis] - rho*vy*vy

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pyz(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vy = get_vy(in_mom)
  _, vz = get_vz(in_mom)
  out_values = in_values[..., 8, np.newaxis] - rho*vy*vz

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pzz(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vz = get_vz(in_mom)
  out_values = in_values[..., 9, np.newaxis] - rho*vz*vz

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_pij(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = np.zeros(in_values[..., 4:10].shape)

  _, pxx = get_pxx(in_mom)
  _, pxy = get_pxy(in_mom)
  _, pxz = get_pxz(in_mom)
  _, pyy = get_pyy(in_mom)
  _, pyz = get_pyz(in_mom)
  _, pzz = get_pzz(in_mom)

  out_values[..., 0] = np.squeeze(pxx)
  out_values[..., 1] = np.squeeze(pxy)
  out_values[..., 2] = np.squeeze(pxz)
  out_values[..., 3] = np.squeeze(pyy)
  out_values[..., 4] = np.squeeze(pyz)
  out_values[..., 5] = np.squeeze(pzz)

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_p(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    num_moms: int | None = None,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  num_comps = in_values.shape[-1]
  if num_moms is None:
    if num_comps == 5:
      num_moms = 5
    elif num_comps == 10:
      num_moms = 10
    else:
      raise ValueError(f"Number of components appears to be {num_comps:d}; it needs to be specified using 'num_moms' (5 or 10)")
    # end
  # end

  if num_moms == 5:
    _, rho = get_density(in_mom)
    _, vx = get_vx(in_mom)
    _, vy = get_vy(in_mom)
    _, vz = get_vz(in_mom)
    out_values = (gas_gamma - 1) * (
        in_values[..., 4, np.newaxis] - 0.5*rho*(vx**2 + vy**2 + vz**2)
    )
  else: # num_moms == 10:
    _, pxx = get_pxx(in_mom)
    _, pyy = get_pyy(in_mom)
    _, pzz = get_pzz(in_mom)
    out_values = (pxx + pyy + pzz) / 3.0
  # end

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_ke(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    num_moms: int | None = None,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  num_comps = in_values.shape[-1]
  if num_moms is None:
    if num_comps == 5:
      num_moms = 5
    elif num_comps == 10:
      num_moms = 10
    else:
      raise ValueError(f"Number of components appears to be {num_comps:d}; (5 or 10)")
    # end
  # end

  if num_moms == 5:
    _, pr = get_p(in_mom, gas_gamma=gas_gamma, num_moms=num_moms)
    out_values = in_values[..., 4, np.newaxis] - pr / (gas_gamma - 1)
  else: #  num_moms == 10:
    _, rho = get_density(in_mom)
    _, vx = get_vx(in_mom)
    _, vy = get_vy(in_mom)
    _, vz = get_vz(in_mom)
    out_values = 0.5*rho*(vx**2 + vy**2 + vz**2)
  # end

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_temp(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    num_moms: int | None = None,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, rho = get_density(in_mom)
  _, pr = get_p(in_mom, gas_gamma=gas_gamma, num_moms=num_moms)
  out_values = pr/rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_sound(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    num_moms: int | None = None,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, rho = get_density(in_mom)
  _, pr = get_p(in_mom, gas_gamma=gas_gamma, num_moms=num_moms)
  out_values = np.sqrt(gas_gamma*pr / rho)

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mach(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    num_moms: int | None = None,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, vx = get_vx(in_mom)
  _, vy = get_vy(in_mom)
  _, vz = get_vz(in_mom)
  _, cs = get_sound(in_mom, gas_gamma=gas_gamma, num_moms=num_moms)
  out_values = np.sqrt(vx**2 + vy**2 + vz**2) / cs

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_Bx(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = in_values[..., 5, np.newaxis]

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_By(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = in_values[..., 6, np.newaxis]

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_Bz(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = in_values[..., 7, np.newaxis]

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_Bi(in_mom: GData | Tuple[list, np.ndarray],
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  out_values = in_values[..., 5:8]

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_mag_p(in_mom: GData | Tuple[list, np.ndarray], mu_0: float = 1.0,
    out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, Bx = get_mhd_Bx(in_mom)
  _, By = get_mhd_By(in_mom)
  _, Bz = get_mhd_Bz(in_mom)
  out_values = 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_p(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    mu_0: float = 1.0, out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, in_values = input_parser(in_mom)
  _, rho = get_density(in_mom)
  _, vx = get_vx(in_mom)
  _, vy = get_vy(in_mom)
  _, vz = get_vz(in_mom)
  _, mag_p = get_mhd_mag_p(in_mom, mu_0=mu_0)

  out_values = (gas_gamma - 1)*(in_values[..., 4, np.newaxis] - 0.5*rho*(vx**2 + vy**2 + vz**2) - mag_p)

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_temp(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    mu_0: float = 1.0, out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, rho = get_density(in_mom)
  _, pr = get_mhd_p(in_mom, gas_gamma=gas_gamma, mu_0=mu_0)
  out_values = pr / rho

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_sound(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    mu_0: float = 1.0, out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, rho = get_density(in_mom)
  _, pr = get_mhd_p(in_mom, gas_gamma=gas_gamma, mu_0=mu_0)

  out_values = np.sqrt(gas_gamma*pr/rho)

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values


def get_mhd_mach(in_mom: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3,
    mu_0: float = 1.0, out_mom: GData | None = None) -> Tuple[list, np.ndarray]:
  grid, vx = get_vx(in_mom)
  _, vy = get_vy(in_mom)
  _, vz = get_vz(in_mom)
  _, cs = get_mhd_sound(in_mom, gas_gamma=gas_gamma, mu_0=mu_0)
  out_values = np.sqrt(vx**2 + vy**2 + vz**2) / cs

  if out_mom:
    out_mom.push(grid, out_values)
  # end
  return grid, out_values
