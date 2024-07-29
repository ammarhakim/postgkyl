import click
import numpy as np


def _get_grid(grid0, grid1):
  if grid0 is not None and grid1 is not None:
    if len(grid0) > len(grid1):
      return grid0
    else:
      return grid1
    # end
  elif grid0 is not None:
    return grid0
  elif grid1 is not None:
    return grid1
  else:
    return None
  # end


def add(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = in_values[0] + in_values[1]
  return [out_grid], [out_values]


def subtract(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = in_values[1] - in_values[0]
  return [out_grid], [out_values]


def mult(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  a, b = in_values[1], in_values[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    out_values = a * b
  else:
    # When multiplying phase-space and conf-space field, the
    # dimensions do not match. NumPy can do a lot of things with
    # broadcasting
    # (https://numpy.org/doc/stable/user/basics.broadcasting.html) but
    # it requires the trailing indices to match, which is opposite to
    # what we have (the first indices are matching). Therefore, one can
    # transpose, multiply, and transpose back... I think -- Petr Cagas
    out_values = (a.transpose() * b.transpose()).transpose()
  # end
  return [out_grid], [out_values]


def dot(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.sum(in_values[1] * in_values[0], axis=-1)[..., np.newaxis]
  return [out_grid], [out_values]


def divide(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  a, b = in_values[1], in_values[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    out_values = a/b
  else:
    # See the 'mult' comment above
    out_values = (a.transpose()/b.transpose()).transpose()
  # end
  return [out_grid], [out_values]


def sqrt(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.sqrt(in_values[0])
  return [out_grid], [out_values]


def psin(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.sin(in_values[0])
  return [out_grid], [out_values]


def pcos(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.cos(in_values[0])
  return [out_grid], [out_values]


def ptan(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.tan(in_values[0])
  return [out_grid], [out_values]


def absolute(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.abs(in_values[0])
  return [out_grid], [out_values]


def log(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.log(in_values[0])
  return [out_grid], [out_values]


def log10(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.log10(in_values[0])
  return [out_grid], [out_values]


def minimum(in_grid, in_values):
  out_values = np.atleast_1d(np.nanmin(in_values[0]))
  return [[]], [out_values]


def minimum2(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.fmin(in_values[0], in_values[1])
  return [out_grid], [out_values]


def maximum(in_grid, in_values):
  out_values = np.atleast_1d(np.nanmax(in_values[0]))
  return [[]], [out_values]


def maximum2(in_grid, in_values):
  out_grid = _get_grid(in_grid[0], in_grid[1])
  out_values = np.fmax(in_values[0], in_values[1])
  return [out_grid], [out_values]


def mean(in_grid, in_values):
  out_values = np.atleast_1d(np.mean(in_values[0]))
  return [[]], [out_values]


def power(in_grid, in_values):
  out_grid = in_grid[1]
  out_values = np.power(in_values[1], in_values[0])
  return [out_grid], [out_values]


def sq(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = in_values[0]**2
  return [out_grid], [out_values]


def exp(in_grid, in_values):
  out_grid = in_grid[0]
  out_values = np.exp(in_values[0])
  return [out_grid], [out_values]


def length(in_grid, in_values):
  ax = int(in_values[0])
  ln = in_grid[1][ax][-1] - in_grid[1][ax][0]
  if len(in_grid[1][ax]) == in_values[1].shape[ax]:
    ln += in_grid[1][ax][1] - in_grid[1][ax][0]
  # end
  return [[]], [ln]


def grad(in_grid, in_values):
  out_grid = in_grid[0]
  nd = len(in_values[0].shape) - 1
  out_shape = list(in_values[0].shape)
  nc = in_values[0].shape[-1]
  out_shape[-1] = nc * nd
  out_values = np.zeros(out_shape)

  for d in range(nd):
    zc = 0.5 * (in_grid[0][d][1:] + in_grid[0][d][:-1])  # get cell centered values
    out_values[..., d*nc:(d + 1)*nc] = np.gradient(
        in_values[0], zc, edge_order=2, axis=d
    )
  # end
  return [out_grid], [out_values]


def grad2(in_grid, in_values):
  out_grid = in_grid[1]
  ax = in_values[0]
  if isinstance(ax, str) and ":" in ax:
    tmp = ax.split(":")
    lo = int(tmp[0])
    up = int(tmp[1])
    rng = range(lo, up)
  elif isinstance(ax, str):
    rng = tuple((int(i) for i in ax.split(",")))
  else:
    rng = range(int(ax), int(ax + 1))
  # end

  num_dims = len(rng)
  out_shape = list(in_values[1].shape)
  num_comps = in_values[1].shape[-1]
  out_shape[-1] = out_shape[-1] * num_dims
  out_values = np.zeros(out_shape)

  for cnt, d in enumerate(rng):
    zc = 0.5 * (in_grid[1][d][1:] + in_grid[1][d][:-1])  # get cell centered values
    out_values[..., cnt*num_comps:(cnt + 1)*num_comps] = np.gradient(
        in_values[1], zc, edge_order=2, axis=d
    )
  # end
  return [out_grid], [out_values]


def integrate(in_grid, in_values, avg=False):
  grid = in_grid[1].copy()
  values = np.array(in_values[1])

  axis = in_values[0]
  if isinstance(axis, float):
    axis = tuple([int(axis)])
  elif isinstance(axis, tuple):
    pass
  elif isinstance(axis, np.ndarray):
    axis = tuple([int(axis)])
  elif isinstance(axis, str):
    if len(axis.split(",")) > 1:
      axes = axis.split(",")
      axis = tuple([int(a) for a in axes])
    elif len(axis.split(":")) == 2:
      bounds = axis.split(":")
      axis = tuple(range(bounds[0], bounds[1]))
    elif axis == "all":
      num_dims = len(grid)
      axis = tuple(range(num_dims))
    # end
  else:
    raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")
  # end

  dz = []
  for d, coord in enumerate(grid):
    dz.append(coord[1:] - coord[:-1])
    if len(coord) == values.shape[d]:
      dz[-1] = np.append(dz[-1], dz[-1][-1])
    # end

  # Integration assuming values are cell centered averages
  # Should work for nonuniform meshes
  for ax in sorted(axis, reverse=True):
    values = np.moveaxis(values, ax, -1)
    values = np.dot(values, dz[ax])
  # end
  for ax in sorted(axis):
    grid[ax] = np.array([0])
    values = np.expand_dims(values, ax)
    if avg:
      ln = in_grid[1][ax][-1] - in_grid[1][ax][0]
      if len(in_grid[1][ax]) == in_values[1].shape[ax]:
        ln += in_grid[1][ax][1] - in_grid[1][ax][0]
      # end
      values = values/ln
    # end
  # end
  return [grid], [values]


def average(in_grid, in_values):
  return integrate(in_grid, in_values, True)


def divergence(in_grid, in_values):
  out_grid = in_grid[0]
  num_dims = len(in_grid[0])
  num_comps = in_values[0].shape[-1]
  if num_comps > num_dims:
    click.echo(
        click.style(f"WARNING in 'ev div': Length of the provided vector ({num_comps:d}) is longer than number of dimensions ({num_dims:d}). The last {num_comps - num_dims:d} component(s) of the vector will be disregarded.",
            fg="yellow")
    )
    # end
  out_shape = list(in_values[0].shape)
  out_shape[-1] = 1
  out_values = np.zeros(out_shape)
  for d in range(num_dims):
    zc = 0.5 * (in_grid[0][d][1:] + in_grid[0][d][:-1])  # get cell centered values
    out_values[..., 0] = out_values[..., 0] + np.gradient(
        in_values[0][..., d], zc, edge_order=2, axis=d
    )
  # end
  return [out_grid], [out_values]


def curl(in_grid, in_values):
  out_grid = in_grid[0]
  num_dims = len(in_grid[0])
  num_comps = in_values[0].shape[-1]

  out_shape = list(in_values[0].shape)

  if num_dims == 1:
    if num_comps != 3:
      raise ValueError(f"ERROR in 'ev curl': Curl in 1D requires 3-component input and {num_comps:d}-component field was provided.")
    # end
    zc0 = 0.5*(in_grid[0][0][1:] + in_grid[0][0][:-1])
    out_values = np.zeros(out_shape)
    out_values[..., 1] = -np.gradient(in_values[0][..., 2], zc0, edge_order=2, axis=0)
    out_values[..., 2] = np.gradient(in_values[0][..., 1], zc0, edge_order=2, axis=0)
  elif num_dims == 2:
    zc0 = 0.5 * (in_grid[0][0][1:] + in_grid[0][0][:-1])
    zc1 = 0.5 * (in_grid[0][1][1:] + in_grid[0][1][:-1])
    if num_comps < 2:
      raise ValueError(f"ERROR in 'ev curl': Length of the provided vector ({num_comps:d}) is smaller than number of dimensions ({num_dims:d}). Curl can't be calculated." )
    elif num_comps == 2:
      click.echo(
          click.style(f"WARNING in 'ev curl': Length of the provided vector ({num_comps:d}) is longer than number of dimensions ({num_dims:d}). Only the third component of curl will be calculated.",
              fg="yellow")
      )
      out_shape[-1] = 1
      out_values = np.zeros(out_shape)
      out_values[..., 0] = np.gradient(
          in_values[0][..., 1], zc0, edge_order=2, axis=0
      ) - np.gradient(in_values[0][..., 0], zc1, edge_order=2, axis=1)
    else:
      if num_comps > 3:
        print("here")
        click.echo(
            click.style(f"WARNING in 'ev curl': Length of the provided vector ({num_comps:d}) is longer than number of dimensions ({num_dims:d}). The last {num_comps - num_dims:d} components of the vector will be disregarded.",
                fg="yellow")
        )
      # end
      out_values = np.zeros(out_shape)
      out_values[..., 0] = np.gradient(in_values[0][..., 2], zc1, edge_order=2, axis=1)
      out_values[..., 1] = -np.gradient(in_values[0][..., 2], zc0, edge_order=2, axis=0)
      out_values[..., 2] = np.gradient( in_values[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(in_values[0][..., 0], zc1, edge_order=2, axis=1)
  else:  # 3D
    if num_comps > 3:
      click.echo(
          click.style(f"WARNING in 'ev curl': Length of the provided vector ({num_comps:d}) is longer than number of dimensions ({num_dims:d}). The last {num_comps - num_dims:d} component(s) of the vector will be disregarded.",
              fg="yellow")
      )
    elif num_comps < 3:
      raise ValueError(
          f"ERROR in 'ev curl': Length of the provided vector ({num_comps:d}) is smaller than number of dimensions ({num_dims:d}). Curl can't be calculated."
      )
    # end
    zc0 = 0.5 * (in_grid[0][0][1:] + in_grid[0][0][:-1])
    zc1 = 0.5 * (in_grid[0][1][1:] + in_grid[0][1][:-1])
    zc2 = 0.5 * (in_grid[0][2][1:] + in_grid[0][2][:-1])
    out_values[..., 0] = np.gradient(in_values[0][..., 2], zc1, edge_order=2, axis=1) - np.gradient(in_values[0][..., 1], zc2, edge_order=2, axis=2)
    out_values[..., 1] = np.gradient(in_values[0][..., 0], zc2, edge_order=2, axis=2) - np.gradient(in_values[0][..., 2], zc0, edge_order=2, axis=0)
    out_values[..., 2] = np.gradient(in_values[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(in_values[0][..., 0], zc1, edge_order=2, axis=1)
  # end
  return [out_grid], [out_values]


cmds = {
    "+": {"num_in": 2, "num_out": 1, "func": add},
    "-": {"num_in": 2, "num_out": 1, "func": subtract},
    "*": {"num_in": 2, "num_out": 1, "func": mult},
    "/": {"num_in": 2, "num_out": 1, "func": divide},
    "dot": {"num_in": 2, "num_out": 1, "func": dot},
    "sqrt": {"num_in": 1, "num_out": 1, "func": sqrt},
    "sin": {"num_in": 1, "num_out": 1, "func": psin},
    "cos": {"num_in": 1, "num_out": 1, "func": pcos},
    "tan": {"num_in": 1, "num_out": 1, "func": ptan},
    "abs": {"num_in": 1, "num_out": 1, "func": absolute},
    "avg": {"num_in": 2, "num_out": 1, "func": average},
    "log": {"num_in": 1, "num_out": 1, "func": log},
    "log10": {"num_in": 1, "num_out": 1, "func": log10},
    "max": {"num_in": 1, "num_out": 1, "func": maximum},
    "min": {"num_in": 1, "num_out": 1, "func": minimum},
    "max2": {"num_in": 2, "num_out": 1, "func": maximum2},
    "min2": {"num_in": 2, "num_out": 1, "func": minimum2},
    "mean": {"num_in": 1, "num_out": 1, "func": mean},
    "len": {"num_in": 2, "num_out": 1, "func": length},
    "pow": {"num_in": 2, "num_out": 1, "func": power},
    "sq": {"num_in": 1, "num_out": 1, "func": sq},
    "exp": {"num_in": 1, "num_out": 1, "func": exp},
    "grad": {"num_in": 1, "num_out": 1, "func": grad},
    "grad2": {"num_in": 2, "num_out": 1, "func": grad2},
    "int": {"num_in": 2, "num_out": 1, "func": integrate},
    "div": {"num_in": 1, "num_out": 1, "func": divergence},
    "curl": {"num_in": 1, "num_out": 1, "func": curl},
}
