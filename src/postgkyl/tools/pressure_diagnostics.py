#!/usr/bin/env python
"""
Postgkyl module for pressure tensor diagnostics
Diagnostics include:
	Pressure parallel to the magnetic field
	Pressure perpendicular to the magnetic field
	Agyrotropy (either Frobenius or Swisdak measure)
	Firehose instability threshold
"""
import numpy as np
import postgkyl.tools as diag


def _get_pb(
    p_data=None, p_grid=None, p_values=None, b_data=None, b_grid=None, b_values=None
):
  if p_data:
    p_grid = p_data.get_grid()
    p_values = p_data.get_values()
  # end
  if b_data:
    b_grid = b_data.get_grid()
    b_values = b_data.get_values()
  # end

  p_xx = p_values[..., 0, np.newaxis]
  p_xy = p_values[..., 1, np.newaxis]
  p_xz = p_values[..., 2, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_yz = p_values[..., 4, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  b_x = b_values[..., 0, np.newaxis]
  b_y = b_values[..., 1, np.newaxis]
  b_z = b_values[..., 2, np.newaxis]
  return p_xx, p_xy, p_xz, p_yy, p_yz, p_zz, b_x, b_y, b_z


# end


def _get_sf(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
):
  if species_data:
    species_grid = species_data.get_grid()
    species_values = species_data.get_values()
  # end
  if field_data:
    field_grid = field_data.get_grid()
    field_values = field_data.get_values()
  # end

  p_grid, p_values = diag.get_pij(species_grid, species_values)
  b_grid = p_grid
  b_values = field_values[..., 3:6]
  return p_grid, p_values, b_grid, b_values


# end


def get_p_par(
    p_data=None, p_grid=None, p_values=None, b_data=None, b_grid=None, b_values=None
):
  if p_data:
    p_grid = p_data.get_grid()
    p_values = p_data.get_values()
  # end
  if b_data:
    b_grid = b_data.get_grid()
    b_values = b_data.get_values()
  # end

  p_xx = p_values[..., 0, np.newaxis]
  p_xy = p_values[..., 1, np.newaxis]
  p_xz = p_values[..., 2, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_yz = p_values[..., 4, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  b_x = b_values[..., 0, np.newaxis]
  b_y = b_values[..., 1, np.newaxis]
  b_z = b_values[..., 2, np.newaxis]

  grid, mag_b_sq = diag.mag_sq(in_grid=b_grid, in_values=b_values)

  out = (
      b_x * b_x * p_xx
      + b_y * b_y * p_yy
      + b_z * b_z * p_zz
      + 2.0 * (b_x * b_y * p_xy + b_x * b_z * p_xz + b_y * b_z * p_yz)
  ) / mag_b_sq
  return grid, out


# end


def get_gkyl_10m_p_par(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
):
  if species_data:
    species_grid = species_data.get_grid()
    species_values = species_data.get_values()
  # end
  if field_data:
    field_grid = field_data.get_grid()
    field_values = field_data.get_values()
  # end

  p_grid, p_values = diag.get_pij(in_grid=species_grid, in_values=species_values)
  b_grid = p_grid
  b_values = field_values[..., 3:6]

  return get_p_par(p_grid=p_grid, p_values=p_values, b_grid=b_grid, b_values=b_values)


# end


def get_p_perp(
    p_data=None, p_grid=None, p_values=None, b_data=None, b_grid=None, b_values=None
):
  if p_data:
    p_grid = p_data.get_grid()
    p_values = p_data.get_values()
  # end
  if b_data:
    b_grid = b_data.get_grid()
    b_values = b_data.get_values()
  # end

  p_xx = p_values[..., 0, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  grid, p_par = get_p_par(
      p_grid=p_grid, p_values=p_values, b_grid=b_grid, b_values=b_values
  )

  out = (p_xx + p_yy + p_zz - p_par) / 2.0
  return grid, out


# end


def get_gkyl_10m_p_perp(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
):
  if species_data:
    species_grid = species_data.get_grid()
    species_values = species_data.get_values()
  # end
  if field_data:
    field_grid = field_data.get_grid()
    field_values = field_data.get_values()
  # end

  p_grid, p_values = diag.get_pij(in_grid=species_grid, in_values=species_values)
  b_grid = p_grid
  b_values = field_values[..., 3:6]

  return get_p_perp(p_grid=p_grid, p_values=p_values, b_grid=b_grid, b_values=b_values)


# end


def get_agyro(
    p_data=None,
    p_grid=None,
    p_values=None,
    b_data=None,
    b_grid=None,
    b_values=None,
    measure="swisdak",
):
  if p_data:
    p_grid = p_data.get_grid()
    p_values = p_data.get_values()
  # end
  if b_data:
    b_grid = b_data.get_grid()
    b_values = b_data.get_values()
  # end

  p_xx = p_values[..., 0, np.newaxis]
  p_xy = p_values[..., 1, np.newaxis]
  p_xz = p_values[..., 2, np.newaxis]
  p_yy = p_values[..., 3, np.newaxis]
  p_yz = p_values[..., 4, np.newaxis]
  p_zz = p_values[..., 5, np.newaxis]

  b_x = b_values[..., 0, np.newaxis]
  b_y = b_values[..., 1, np.newaxis]
  b_z = b_values[..., 2, np.newaxis]

  grid, mag_b_sq = diag.mag_sq(in_grid=b_grid, in_values=b_values)
  _, p_par = get_p_par(
      p_grid=p_grid, p_values=p_values, b_grid=b_grid, b_values=b_values
  )
  _, p_perp = get_p_perp(
      p_grid=p_grid, p_values=p_values, b_grid=b_grid, b_values=b_values
  )

  if measure.lower() == "swisdak":
    I1 = p_xx + p_yy + p_zz
    I2 = (
        p_xx * p_yy
        + p_xx * p_zz
        + p_yy * p_zz
        - (p_xy * p_xy + p_xz * p_xz + p_yz * p_yz)
    )

    # Note that this definition of Q uses the tensor algebra in
    # Appendix A of Swisdak 2015.
    out = np.sqrt(1 - 4 * I2 / ((I1 - p_par) * (I1 + 3 * p_par)))
  elif measure.lower() == "frobenius":
    p_ixx = p_xx - (p_par * b_x * b_x / mag_b_sq + p_perp * (1 - b_x * b_x / mag_b_sq))
    p_ixy = p_xy - (p_par * b_x * b_y / mag_b_sq + p_perp * (0 - b_x * b_y / mag_b_sq))
    p_ixz = p_xz - (p_par * b_x * b_z / mag_b_sq + p_perp * (0 - b_x * b_z / mag_b_sq))
    p_iyy = p_yy - (p_par * b_y * b_y / mag_b_sq + p_perp * (1 - b_y * b_y / mag_b_sq))
    p_iyz = p_yz - (p_par * b_y * b_z / mag_b_sq + p_perp * (0 - b_y * b_z / mag_b_sq))
    p_izz = p_zz - (p_par * b_z * b_z / mag_b_sq + p_perp * (1 - b_z * b_z / mag_b_sq))
    out = np.sqrt(
        p_ixx**2 + 2 * p_ixy**2 + 2 * p_ixz**2 + p_iyy**2 + 2 * p_iyz**2 + p_izz**2
    ) / np.sqrt(2 * p_perp**2 + 4 * p_par * p_perp)
  else:
    raise ValueError(
        "Measure specified is {:s};"
        "it needs to be either 'swisdak' or 'frobenius'".format(measure.lower())
    )
  # end
  return grid, out


# end


def get_gkyl_10m_agyro(
    species_data=None,
    species_grid=None,
    species_values=None,
    field_data=None,
    field_grid=None,
    field_values=None,
    measure="swisdak",
):
  if species_data:
    species_grid = species_data.get_grid()
    species_values = species_data.get_values()
  # end
  if field_data:
    field_grid = field_data.get_grid()
    field_values = field_data.get_values()
  # end

  p_grid, p_values = diag.get_pij(in_grid=species_grid, in_values=species_values)
  b_grid = p_grid
  b_values = field_values[..., 3:6]

  return get_agyro(
      p_grid=p_grid,
      p_values=p_values,
      b_grid=b_grid,
      b_values=b_values,
      measure=measure,
  )


# end
