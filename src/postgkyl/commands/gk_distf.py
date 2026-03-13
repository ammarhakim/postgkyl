import os

import click
import numpy as np

from postgkyl.data import GData
from postgkyl.data import GInterpModal
from postgkyl.utils import verb_print


def _resolve_path(path: str, file_name: str) -> str:
  if os.path.isabs(file_name):
    return file_name
  return os.path.join(path, file_name)


def _resolve_files(name: str, species: str, frame: int, path: str, source: bool = False):
  if source:
    frame_file = _resolve_path(path, f"{name}-{species}_source_{frame}.gkyl")
  else:
    frame_file = _resolve_path(path, f"{name}-{species}_{frame}.gkyl")
  mapc2p_vel_file = _resolve_path(path, f"{name}-{species}_mapc2p_vel.gkyl")
  mc2nu_file = _resolve_path(path, f"{name}-mc2nu_pos_deflated.gkyl")
  jacobvel_file = _resolve_path(path, f"{name}-{species}_jacobvel.gkyl")
  jacobtot_inv_file = _resolve_path(path, f"{name}-jacobtot_inv.gkyl")
  return frame_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file


def _resolve_mapping_kwargs(use_c2p_vel: bool, mapc2p_vel_file: str) -> dict:
  if use_c2p_vel:
    return {"mapc2p_vel_name": mapc2p_vel_file}
  return {"mapc2p_vel_name": ""}


def _cell_centers_to_nodes(cell_centers: np.ndarray) -> np.ndarray:
  nodes = np.zeros(cell_centers.size + 1, dtype=cell_centers.dtype)
  nodes[1:-1] = 0.5 * (cell_centers[:-1] + cell_centers[1:])
  nodes[0] = cell_centers[0] - (nodes[1] - cell_centers[0])
  nodes[-1] = cell_centers[-1] + (cell_centers[-1] - nodes[-2])
  return nodes


def _get_dim_split(out_grid: list, mc2nu_data: GData) -> tuple[int, int]:
  total_dims = len(out_grid)
  cdim = mc2nu_data.get_num_dims()
  if cdim < 1 or cdim > total_dims:
    raise ValueError(f"Invalid cdim={cdim} for total_dims={total_dims}")
  vdim = total_dims - cdim
  return cdim, vdim


def _extract_mapped_axis(mapped_values: np.ndarray, axis: int, cdim: int) -> np.ndarray:
  if cdim == 1:
    return np.asarray(mapped_values[..., axis]).reshape(-1)

  # Extract one mapped coordinate direction at a reference point in the
  # remaining configuration directions to construct a 1D axis grid.
  idx = [0] * (cdim + 1)
  idx[axis] = slice(None)
  idx[-1] = axis
  return np.asarray(mapped_values[tuple(idx)]).reshape(-1)


def _apply_mc2nu_grid(out_grid: list, mc2nu_file: str, debug: bool) -> tuple[list, int, int]:
  mc2nu_data = GData(mc2nu_file)
  cdim, vdim = _get_dim_split(out_grid, mc2nu_data)

  mc2nu_interp = GInterpModal(mc2nu_data, 1, "ms")
  _, mc2nu_values = mc2nu_interp.interpolate(tuple(range(cdim)))
  mapped_values = np.asarray(mc2nu_values)

  deformed_grid = list(out_grid)
  for d in range(cdim):
    mapped_cfg = _extract_mapped_axis(mapped_values, d, cdim)

    old_cfg = np.asarray(out_grid[d])
    if mapped_cfg.size == old_cfg.size:
      new_cfg = mapped_cfg
    elif mapped_cfg.size + 1 == old_cfg.size:
      new_cfg = _cell_centers_to_nodes(mapped_cfg)
    else:
      raise ValueError(
          "mc2nu mapping size is incompatible with configuration grid on axis "
          f"{d}: {mapped_cfg.size} vs {old_cfg.size}"
      )

    deformed_grid[d] = new_cfg

  if debug:
    click.echo(f"gk_distf: cdim={cdim}, vdim={vdim}")
    click.echo(f"gk_distf: mc2nu mapped {cdim} configuration axis/axes")

  return deformed_grid, cdim, vdim


def _assert_files_exist(files: dict):
  for kind, file_name in files.items():
    if not os.path.exists(file_name):
      raise FileNotFoundError(f"Missing {kind} file: {file_name}")


def _compute_fjx(jf_values: np.ndarray, jac_values: np.ndarray) -> np.ndarray:
  if jf_values.shape == jac_values.shape:
    return jf_values / jac_values

  if jf_values.shape[:-1] != jac_values.shape[:-1]:
    raise ValueError(
        "Jf and jacobvel spatial shapes do not match: "
        f"{jf_values.shape} vs {jac_values.shape}"
    )

  if jac_values.shape[-1] != 1:
    raise ValueError(
        "jacobvel component axis must be 1 or match Jf: "
        f"{jf_values.shape} vs {jac_values.shape}"
    )

  return jf_values / jac_values


def _broadcast_multiply(distf: np.ndarray, jacobtot_inv: np.ndarray) -> np.ndarray:
  if jacobtot_inv.ndim > distf.ndim:
    raise ValueError(
        "jacobtot_inv has more dimensions than distribution function: "
        f"{jacobtot_inv.ndim} > {distf.ndim}"
    )

  expected_prefix = distf.shape[:jacobtot_inv.ndim]
  if expected_prefix != jacobtot_inv.shape:
    raise ValueError(
        "Cannot broadcast jacobtot_inv over distribution function. "
        f"Expected prefix {expected_prefix}, got {jacobtot_inv.shape}"
    )

  jacob_shape = jacobtot_inv.shape + (1,) * (distf.ndim - jacobtot_inv.ndim)
  return distf * jacobtot_inv.reshape(jacob_shape)


def _interpolate_fjx_and_jacob(fjx_data: GData, jacobtot_inv_data: GData):
  fjx_interp = GInterpModal(fjx_data, 1, "gkhyb")
  jacob_interp = GInterpModal(jacobtot_inv_data, 1, "ms")

  grid, fjx_values = fjx_interp.interpolate()
  _, jacob_values = jacob_interp.interpolate()

  fjx_values = np.squeeze(fjx_values)
  jacob_values = np.squeeze(jacob_values)
  return grid, fjx_values, jacob_values


@click.command()
@click.option("--name", "-n", required=True, type=click.STRING,
    help="Simulation name prefix (e.g. gk_lorentzian_mirror).")
@click.option("--species", "-s", required=True, type=click.STRING,
    help="Species name (e.g. ion or elc).")
@click.option("--frame", "-f", required=True, type=click.INT,
    help="Frame number.")
@click.option("--source", is_flag=True,
  help="Use <name>-<species>_source_<frame>.gkyl as the input distribution.")
@click.option("--path", "-p", default="./", type=click.STRING,
    help="Path to simulation data.")
@click.option("--tag", "-t", default="df", type=click.STRING,
    help="Tag for output dataset.")
@click.option("--c2p-vel/--no-c2p-vel", default=False,
    help="Use <name>-<species>_mapc2p_vel.gkyl when loading Jf.")
@click.option("--mc2nu", is_flag=True,
  help="Use <name>-mc2nu_pos_deflated.gkyl to deform configuration-space grid.")
@click.option("--debug", is_flag=True,
    help="Print resolved file names and shape diagnostics.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  verb_print(ctx, "Starting gk_distf")
  data = ctx.obj["data"]

  files = {}
  jf_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file = _resolve_files(
      kwargs["name"], kwargs["species"], kwargs["frame"], kwargs["path"], kwargs["source"])
  files["Jf"] = jf_file
  files["jacobvel"] = jacobvel_file
  files["jacobtot_inv"] = jacobtot_inv_file

  if kwargs["c2p_vel"]:
    files["mapc2p_vel"] = mapc2p_vel_file
  if kwargs["mc2nu"]:
    files["mc2nu"] = mc2nu_file

  _assert_files_exist(files)

  map_kwargs = _resolve_mapping_kwargs(kwargs["c2p_vel"], mapc2p_vel_file)

  if kwargs["debug"]:
    click.echo(f"gk_distf: Jf={jf_file}")
    click.echo(f"gk_distf: jacobvel={jacobvel_file}")
    click.echo(f"gk_distf: jacobtot_inv={jacobtot_inv_file}")
    if kwargs["c2p_vel"]:
      click.echo(f"gk_distf: mapc2p_vel={mapc2p_vel_file}")
    if kwargs["mc2nu"]:
      click.echo(f"gk_distf: mc2nu={mc2nu_file}")

  jf_data = GData(jf_file, mapc2p_vel_name=map_kwargs["mapc2p_vel_name"])
  jacobvel_data = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  fjx_data = GData(tag=kwargs["tag"], ctx=jf_data.ctx)
  fjx_values = _compute_fjx(jf_data.get_values(), jacobvel_data.get_values())
  fjx_data.push(jf_data.get_grid(), fjx_values)

  out_grid, fjx_interp, jacob_interp = _interpolate_fjx_and_jacob(fjx_data, jacobtot_inv_data)
  distf_values = _broadcast_multiply(fjx_interp, jacob_interp)

  if kwargs["mc2nu"]:
    out_grid, cdim, vdim = _apply_mc2nu_grid(out_grid, mc2nu_file, kwargs["debug"])
    jf_data.ctx["grid_type"] = "mc2nu"

  if kwargs["debug"]:
    click.echo(f"gk_distf: output shape={distf_values.shape}")

  out = GData(tag=kwargs["tag"], ctx=jf_data.ctx)
  out.push(out_grid, np.asarray(distf_values)[..., np.newaxis])
  data.add(out)

  verb_print(ctx, "Finishing gk_distf")