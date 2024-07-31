"""Postgkyl module for plasma related parameters."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import numpy as np

from postgkyl.tools.mag_sq import mag_sq
from postgkyl.tools.prim_vars import get_density, get_temp, get_mhd_temp
from postgkyl.utils import input_parser

if TYPE_CHECKING:
  from postgkyl import GData
# end


def get_magB(field: GData | Tuple[list, np.ndarray]) -> Tuple[list, np.ndarray]:
  field_grid, field_values = input_parser(field)
  b_values = field_values[..., 3:6]
  _, mag_B_sq = mag_sq((field_grid, b_values))
  out_values = np.sqrt(mag_B_sq)

  return field_grid, out_values


def get_vt(species: GData | Tuple[list, np.ndarray], gas_gamma: float = 5.0/3.0,
    num_moms : int | None = None, mass: float = 1.0, mu_0: float = 1.0,
    sqrt2: bool = True, mhd: bool = False) -> Tuple[list, np.ndarray]:
  m = species.ctx["mass"] if species.ctx["mass"] else mass

  if mhd:
    out_grid, temp = get_mhd_temp(species, gas_gamma=gas_gamma, mu_0=mu_0)
  else:
    out_grid, temp = get_temp(species, gas_gamma=gas_gamma, num_moms=num_moms)
  # end
  out_values = np.sqrt(temp/m)
  if sqrt2:
    out_values *= np.sqrt(2.0)

  return out_grid, out_values


def get_vA(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    mu_0: float = 1.0) -> Tuple[list, np.ndarray]:
  mu = field.ctx["mu_0"] if field.ctx["mu_0"] else mu_0

  _, magB = get_magB(field)
  # Fluid data already has mass factor in density
  out_grid, rho = get_density(species)
  out_values = magB/np.sqrt(mu*rho)

  return out_grid, out_values


def get_omegaC(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    mass: float = 1.0, charge: float = 1.0) -> Tuple[list, np.ndarray]:
  m = species.ctx["mass"] if species.ctx["mass"] else mass
  q = species.ctx["charge"] if species.ctx["charge"] else charge

  out_grid, magB = get_magB(field)
  out_values = abs(q)*magB/m

  return out_grid, out_values


def get_omegaP(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0) -> Tuple[list, np.ndarray]:
  m = species.ctx["mass"] if species.ctx["mass"] else mass
  q = species.ctx["charge"] if species.ctx["charge"] else charge
  epsilon = field.ctx["epsilon_0"] if field.ctx["epsilon_0"] else epsilon_0

  # Fluid data already has mass factor in density
  out_grid, rho = get_density(species)
  qbym2 = q**2/m**2
  out_values = np.sqrt(qbym2*rho/epsilon)

  return out_grid, out_values


def get_d(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    mu_0 : float = 1.0) -> Tuple[list, np.ndarray]:
  epsilon = field.ctx["epsilon_0"] if field.ctx["epsilon_0"] else epsilon_0
  mu = field.ctx["mu_0"] if field.ctx["mu_0"] else mu_0

  out_grid, omegaP = get_omegaP(species=species, field=field, mass=mass, charge=charge,
    epsilon_0=epsilon_0)
  light_speed = 1.0/np.sqrt(epsilon*mu)
  out_values = light_speed/omegaP

  return out_grid, out_values


def get_lambdaD(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    gas_gamma: float = 5.0/3.0, num_moms: int | None = None,
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    mu_0 : float = 1.0, sqrt2: float = True) -> Tuple[list, np.ndarray]:
  _, omegaP = get_omegaP(species=species, field=field, mass=mass, charge=charge,
    epsilon_0=epsilon_0)
  out_grid, vt = get_vt(species=species, gas_gamma=gas_gamma, num_moms=num_moms,
      mass=mass, mu_0=mu_0, sqrt2=sqrt2)
  out_values = vt / omegaP
  if sqrt2:
    out_values /= np.sqrt(2.0)
  # end

  return out_grid, out_values


def get_rho(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    gas_gamma: float = 5.0/3.0, num_moms: int | None = None,
    mass: float = 1.0, charge: float = 1.0, epsilon_0: float = 1.0,
    mu_0 : float = 1.0, sqrt2: float = True) -> Tuple[list, np.ndarray]:

  _, omegaC = get_omegaC(species=species, field=field, mass=mass, charge=charge)
  out_grid, vt = get_vt(species=species, gas_gamma=gas_gamma, num_moms=num_moms,
      mass=mass, mu_0=mu_0, sqrt2=sqrt2)

  out_values = vt/omegaC
  if not sqrt2:
    out_values *= np.sqrt(2.0)
  # end

  return out_grid, out_values


def get_beta(species: GData | Tuple[list, np.ndarray], field: GData | Tuple[list, np.ndarray],
    gas_gamma: float = 5.0/3.0, num_moms: int | None = None,
    mass: float = 1.0, mu_0 : float = 1.0, sqrt2: float = True) -> Tuple[list, np.ndarray]:
  _, v_A = get_vA(species=species, field=field, mu_0=mu_0)
  out_grid, vt = get_vt(species=species, gas_gamma=gas_gamma, num_moms=num_moms,
      mass=mass, mu_0=mu_0, sqrt2=sqrt2)
  out_values = vt**2 / v_A**2
  if not sqrt2:
    out_values *= 2.0

  return out_grid, out_values
