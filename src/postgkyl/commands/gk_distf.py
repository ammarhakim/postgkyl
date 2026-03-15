import glob

import click
import numpy as np

from postgkyl.data import GData, GInterpModal


def _parse_file_names(
    name: str, species: str, frame: int,
    source: bool = False, block_idx: int | None = None,
    use_c2p_vel: bool = False, use_mc2nu: bool = False
) -> tuple[str, str, str, str, str]:
  "Find all relevant files for the given frame, checking for existence and returning paths."
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = "source_" if source else ""
  frame_file = f"{prefix}-{species}_{frame_infix}{frame}.gkyl"
  mapc2p_vel_file = f"{prefix}-{species}_mapc2p_vel.gkyl"
  mc2nu_file = f"{prefix}-mc2nu_pos_deflated.gkyl"
  jacobvel_file = f"{prefix}-{species}_jacobvel.gkyl"
  jacobtot_inv_file = f"{prefix}-jacobtot_inv.gkyl"

  # Check all files exist.
  for file in [frame_file, jacobvel_file, jacobtot_inv_file]:
    if not glob.glob(file):
      raise FileNotFoundError(f"Required file not found: {file}")
    # end
  # end
  if use_c2p_vel:
    if not glob.glob(mapc2p_vel_file):
      raise FileNotFoundError(f"Required file not found: {mapc2p_vel_file}")
    # end
  # end

  if use_mc2nu:
    if not glob.glob(mc2nu_file):
      raise FileNotFoundError(f"Required file not found: {mc2nu_file}")
    # end
  # end
  return frame_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file
# end

def _parse_frames_in_directory(
    name: str, species: str, frame: str,
    source: bool = False, block_idx: int | None = None,
) -> list[int]:
  """Parse frame input and resolve to a list of available frame numbers."""
  frame = frame.strip()

  # If frame is a single number, return it as a list.
  if ":" not in frame:
    return [int(frame)]
  # end

  prefix_name = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = "source_" if source else ""
  prefix = f"{prefix_name}-{species}_{frame_infix}"

  # Keep only files whose suffix is an integer frame number.
  frames = [
      int(suffix[:-5])
      for file_name in glob.glob(f"{glob.escape(prefix)}*.gkyl")
      if (suffix := file_name.removeprefix(prefix)).endswith(".gkyl")
      and suffix[:-5].isdigit()
  ]
  available = sorted(set(frames))

  parts = frame.split(":")
  start = int(parts[0]) if parts[0] else None
  stop = int(parts[1]) if parts[1] else None
  step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
  lower = available[0] if start is None else start
  upper = available[-1] + 1 if stop is None else stop

  frames = [
      f for f in available
      if lower <= f < upper and (f - lower) % step == 0
  ]
  return frames
# end

# ---------------------------------------------------------------------------
# mc2nu grid deformation helpers
# ---------------------------------------------------------------------------

def _cell_centers_to_nodes(cell_centers: np.ndarray) -> np.ndarray:
  """ Given an array of cell centers, return the corresponding node coordinates by extrapolating half a cell width at the boundaries."""
  nodes = np.zeros(cell_centers.size + 1, dtype=cell_centers.dtype)
  nodes[1:-1] = 0.5 * (cell_centers[:-1] + cell_centers[1:])
  nodes[0]  = cell_centers[0]  - (nodes[1]  - cell_centers[0])
  nodes[-1] = cell_centers[-1] + (cell_centers[-1] - nodes[-2])
  return nodes
# end


def _extract_mapped_axis(mapped_values: np.ndarray, axis: int, cdim: int) -> np.ndarray:
  """Extract a 1D mapped coordinate array for the given configuration axis."""
  if cdim == 1:
    return np.asarray(mapped_values[..., axis]).reshape(-1)
  # end

  # Pick a reference point in all other config directions.
  idx = [0] * (cdim + 1)
  idx[axis] = slice(None)
  idx[-1] = axis
  return np.asarray(mapped_values[tuple(idx)]).reshape(-1)
# end


def _apply_mc2nu_grid(out_grid: list, mc2nu_file: str, ctx=None) -> list:
  """Replace computational configuration-space grid with non-uniform spatial coordinates."""
  mc2nu_data = GData(mc2nu_file)
  cdim = mc2nu_data.get_num_dims()

  mc2nu_interp = GInterpModal(mc2nu_data, 1, "ms")
  _, mc2nu_values = mc2nu_interp.interpolate(tuple(range(cdim)))
  mapped_values = np.asarray(mc2nu_values)

  deformed_grid = list(out_grid)
  for d in range(cdim):
    mapped_cfg = _extract_mapped_axis(mapped_values, d, cdim)
    deformed_grid[d] = _cell_centers_to_nodes(mapped_cfg)
    # end
  # end
  return deformed_grid
# end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gk_distf(
    name: str, species: str, frame: int,
    tag: str = "df", source: bool = False, use_c2p_vel: bool = False,
    use_mc2nu: bool = False,
    block_idx: int | None = None,
    ctx=None,
) -> GData:
  """Build a real distribution function from saved JBf data."""
  jf_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file = _parse_file_names(
      name, species, frame, source, block_idx=block_idx, use_c2p_vel=use_c2p_vel, use_mc2nu=use_mc2nu)

  jf_data = GData(jf_file, mapc2p_vel_name=mapc2p_vel_file)
  jacobvel_data = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  # Divide Jf by jacobvel to get f * J_x * B
  # Could be improved using weak division
  fjxB_data = GData(tag=tag, ctx=jf_data.ctx)
  fjxB_values = jf_data.get_values() / jacobvel_data.get_values()
  fjxB_data.push(jf_data.get_grid(), fjxB_values)

  # Interpolate f * J_x * B and jacobtot_inv to the same grid.
  out_grid, fjxB_values = GInterpModal(fjxB_data, 1, "gkhyb").interpolate()
  _, jacobtot_inv_values = GInterpModal(jacobtot_inv_data, 1, "ms").interpolate()
  fjxB_values = np.squeeze(fjxB_values)
  jacobtot_inv_values = np.squeeze(jacobtot_inv_values)

  # Reshape jacobtot_inv so that we can multiply f * J_x * B by 1/(J_x B)
  vdim = fjxB_values.ndim - jacobtot_inv_values.ndim
  jacob_shape = jacobtot_inv_values.shape + (1,) * vdim # Single array index in velocity directions
  distf_values = fjxB_values * jacobtot_inv_values.reshape(jacob_shape)

  if use_mc2nu:
    out_grid = _apply_mc2nu_grid(out_grid, mc2nu_file, ctx)
    jf_data.ctx["grid_type"] = "mc2nu"
  # end

  out = GData(tag=tag, ctx=jf_data.ctx)
  out.push(out_grid, np.asarray(distf_values)[..., np.newaxis])
  return out
# end

# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command()
@click.option("--name", "-n", required=True, type=click.STRING,
    help="Simulation name prefix (e.g. gk_lorentzian_mirror).")
@click.option("--species", "-s", required=True, type=click.STRING,
    help="Species name (e.g. ion or elc).")
@click.option("--source", is_flag=True,
    help="Use <name>-<species>_source_<frame>.gkyl as the input distribution.")
@click.option("--frame", "-f", required=True, type=click.STRING,
    help="Frame number or range. Use ':' for all frames and 'start:stop[:step]' for ranges.")
@click.option("--c2p-vel", "-v", is_flag=True, default=False,
    help="Use <name>-<species>_mapc2p_vel.gkyl when loading Jf.")
@click.option("--mc2nu", "-m", is_flag=True,
    help="Use <name>-mc2nu_pos_deflated.gkyl to deform the configuration-space grid.")
@click.option("--block", "-b", default=None, type=click.INT,
  help="Use block-specific files with _b<idx> prefix, e.g. -b 1 loads <name>_b1-*.gkyl.")
@click.option("--tag", "-t", default="f", type=click.STRING,
    help="Tag for output dataset.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  data = ctx.obj["data"]

  frames = _parse_frames_in_directory(
      name=kwargs["name"], species=kwargs["species"],
      frame=kwargs["frame"],
      source=kwargs["source"], block_idx=kwargs["block"])

  common = dict(
      name=kwargs["name"], species=kwargs["species"],
      tag=kwargs["tag"],
      source=kwargs["source"], use_c2p_vel=kwargs["c2p_vel"],
      use_mc2nu=kwargs["mc2nu"],
      block_idx=kwargs["block"],
      ctx=ctx,
  )
  for frame in frames:
    out = load_gk_distf(frame=frame, **common)
    data.add(out)
  # end

  if len(frames) > 1:
    data.set_unique_labels()
  # end
# end