import numpy as np

import postgkyl.tools as diag


def get_magB(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
):
  if field_data:
    field_grid = field_data.get_grid()
    field_values = field_data.get_values()
  # end
  b_grid = field_grid
  b_values = field_values[..., 3:6]
  out_grid, mag_B_sq = diag.mag_sq(in_grid=b_grid, in_values=b_values)
  out_values = np.sqrt(mag_B_sq)

  return out_grid, out_values


def get_vt(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    gasGamma=5.0 / 3.0,
    numMom=None,
    mass=1.0,
    mu0=1.0,
    sqrt2=True,
    mhd=False,
):
  if species_data:
    species_grid = species_data.get_grid()
    species_values = species_data.get_values()
  # end
  if mhd:
    out_grid, temp = diag.get_mhd_temp(species_data, gasGamma, mu0)
  else:
    out_grid, temp = diag.get_temp(species_data, gasGamma, numMom)
  # end

  if species_data.mass:
    _m = species_data.mass
  else:
    _m = mass
  # end

  if sqrt2:
    out_values = np.sqrt(2.0 * temp / _m)
  else:
    out_values = np.sqrt(temp / _m)
  # end

  return out_grid, out_values


def get_vA(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    mass=1.0,
    mu0=1.0,
):

  out_grid, magB = get_magB(
      species_data, species_grid, species_values, field_data, field_grid, field_values
  )

  # Fluid data already has mass factor in density
  out_grid, rho = diag.get_density(species_data, species_grid, species_values)

  if field_data.mu0:
    _mu = mu0
  else:
    _mu = mu0
  # end

  out_values = magB / np.sqrt(_mu * rho)

  return out_grid, out_values


def get_omegaC(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    mass=1.0,
    charge=1.0,
):
  out_grid, magB = get_magB(
      species_data, species_grid, species_values, field_data, field_grid, field_values
  )

  if species_data.mass:
    _m = species_data.mass
  else:
    _m = mass
  # end

  if species_data.charge:
    _q = species_data.charge
  else:
    _q = charge
  # end

  out_values = abs(_q) / _m * magB

  return out_grid, out_values


def get_omegaP(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    mass=1.0,
    charge=1.0,
    epsilon0=1.0,
):
  # Fluid data already has mass factor in density
  out_grid, rho = diag.get_density(species_data, species_grid, species_values)

  if species_data.mass:
    _m = species_data.mass
  else:
    _m = mass
  # end

  if species_data.charge:
    _q = species_data.charge
  else:
    _q = charge
  # end

  if field_data.epsilon0:
    _eps = epsilon0
  else:
    _eps = epsilon0
  # end

  qbym2 = _q * _q / (_m * _m)
  out_values = np.sqrt(qbym2 * rho / _eps)

  return out_grid, out_values


def get_d(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    mass=1.0,
    charge=1.0,
    epsilon0=1.0,
    mu0=1.0,
):

  out_grid, omegaP = get_omegaP(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      mass,
      charge,
      epsilon0,
  )

  if field_data.epsilon0:
    _eps = epsilon0
  else:
    _eps = epsilon0
  # end

  if field_data.mu0:
    _mu = mu0
  else:
    _mu = mu0
  # end

  light_speed = 1.0 / np.sqrt(_eps * _mu)
  out_values = light_speed / omegaP

  return out_grid, out_values


def get_lambdaD(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    gasGamma=5.0 / 3.0,
    numMom=None,
    mass=1.0,
    charge=1.0,
    epsilon0=1.0,
    mu0=1.0,
    sqrt2=True,
):

  out_grid, omegaP = get_omegaP(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      mass,
      charge,
      epsilon0,
  )
  out_grid, vt = get_vt(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      gasGamma,
      numMom,
      mass,
      mu0,
      sqrt2,
  )

  if sqrt2:
    out_values = (vt / omegaP) / np.sqrt(2.0)
  else:
    out_values = vt / omegaP

  return out_grid, out_values


def get_rho(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    gasGamma=5.0 / 3.0,
    numMom=None,
    mass=1.0,
    charge=1.0,
    epsilon0=1.0,
    mu0=1.0,
    sqrt2=True,
):

  out_grid, omegaC = get_omegaC(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      mass,
      charge,
  )
  out_grid, vt = get_vt(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      gasGamma,
      numMom,
      mass,
      mu0,
      sqrt2,
  )

  if sqrt2:
    out_values = vt / omegaC
  else:
    out_values = (vt / omegaC) * np.sqrt(2.0)

  return out_grid, out_values


def get_beta(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    gasGamma=5.0 / 3.0,
    numMom=None,
    mass=1.0,
    charge=1.0,
    epsilon0=1.0,
    mu0=1.0,
    sqrt2=True,
):

  out_grid, vA = get_vA(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      mass,
      mu0,
  )
  out_grid, vt = get_vt(
      species_data,
      species_grid,
      species_values,
      field_data,
      field_grid,
      field_values,
      gasGamma,
      numMom,
      mass,
      mu0,
      sqrt2,
  )

  if sqrt2:
    out_values = vt * vt / (vA * vA)
  else:
    out_values = 2.0 * vt * vt / (vA * vA)

  return out_grid, out_values
