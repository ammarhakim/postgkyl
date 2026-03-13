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


def _resolve_files(name: str, species: str, frame: int, path: str):
  frame_file = _resolve_path(path, f"{name}-{species}_{frame}.gkyl")
  mapc2p_vel_file = _resolve_path(path, f"{name}-{species}_mapc2p_vel.gkyl")
  jacobvel_file = _resolve_path(path, f"{name}-{species}_jacobvel.gkyl")
  jacobtot_inv_file = _resolve_path(path, f"{name}-jacobtot_inv.gkyl")
  return frame_file, mapc2p_vel_file, jacobvel_file, jacobtot_inv_file


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
@click.option("--path", "-p", default="./", type=click.STRING,
    help="Path to simulation data.")
@click.option("--tag", "-t", default="df", type=click.STRING,
    help="Tag for output dataset.")
@click.option("--c2p-vel", default=False,
    help="Use <name>-<species>_mapc2p_vel.gkyl when loading Jf.")
@click.option("--debug", is_flag=True,
    help="Print resolved file names and shape diagnostics.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  verb_print(ctx, "Starting gk_distf")
  data = ctx.obj["data"]

  files = {}
  jf_file, mapc2p_vel_file, jacobvel_file, jacobtot_inv_file = _resolve_files(
      kwargs["name"], kwargs["species"], kwargs["frame"], kwargs["path"])
  files["Jf"] = jf_file
  files["jacobvel"] = jacobvel_file
  files["jacobtot_inv"] = jacobtot_inv_file
  if kwargs["c2p_vel"]:
    files["mapc2p_vel"] = mapc2p_vel_file

  _assert_files_exist(files)

  if kwargs["debug"]:
    click.echo(f"gk_distf: Jf={jf_file}")
    click.echo(f"gk_distf: jacobvel={jacobvel_file}")
    click.echo(f"gk_distf: jacobtot_inv={jacobtot_inv_file}")
    if kwargs["c2p_vel"]:
      click.echo(f"gk_distf: mapc2p_vel={mapc2p_vel_file}")

  jf_data = GData(jf_file, mapc2p_vel_name=mapc2p_vel_file if kwargs["c2p_vel"] else "")
  jacobvel_data = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  fjx_data = GData(tag=kwargs["tag"], ctx=jf_data.ctx)
  fjx_values = _compute_fjx(jf_data.get_values(), jacobvel_data.get_values())
  fjx_data.push(jf_data.get_grid(), fjx_values)

  out_grid, fjx_interp, jacob_interp = _interpolate_fjx_and_jacob(fjx_data, jacobtot_inv_data)
  distf_values = _broadcast_multiply(fjx_interp, jacob_interp)

  if kwargs["debug"]:
    click.echo(f"gk_distf: output shape={distf_values.shape}")

  out = GData(tag=kwargs["tag"], ctx=jf_data.ctx)
  out.push(out_grid, np.asarray(distf_values)[..., np.newaxis])
  data.add(out)

  verb_print(ctx, "Finishing gk_distf")