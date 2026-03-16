import glob

import click
import numpy as np

from postgkyl.data import GData, GInterpModal
from postgkyl.utils import verb_print

# mc2nu grid deformation helpers
# This is a result of the gkyl_reader not having support for both mapc2p and mapc2p-vel grids.
# Particularly, the gkyl_reader does not support mapping phase space arrays with mapc2p
def _convert_cell_centered_to_nodal(cell_centers: np.ndarray) -> np.ndarray:
  """ Given an array defined at cell centers, return the corresponding nodal values
   by interpolating half a cell width at the boundaries."""
  nodes = np.zeros(cell_centers.size + 1, dtype=cell_centers.dtype)
  nodes[1:-1] = 0.5 * (cell_centers[:-1] + cell_centers[1:])
  nodes[0]  = cell_centers[0]  + (cell_centers[0]  - nodes[1]) # Cell center plus half a cell width
  nodes[-1] = cell_centers[-1] + (cell_centers[-1] - nodes[-2]) # Cell center plus half a cell width
  return nodes
# end

def _extract_values_along_dimension(mapped_values: np.ndarray, axis: int, cdim: int) -> np.ndarray:
  """Decompose mapped_values into a 1D array along the specified axis"""
  idx = [0] * (cdim + 1)  # Initialize indexing array. mc2nu has cdim+1 dimensions.
  idx[axis] = slice(None)  # Define a slice along the desired axis.
  idx[-1] = axis  # Select the appropriate component of mc2nu
  return mapped_values[tuple(idx)].reshape(-1)  # Apply indices and flatten to 1D.
# end

def _apply_mc2nu_grid(uniform_grid: list, mc2nu_file: str) -> list:
  """Replace computational configuration-space grid with non-uniform spatial coordinates."""
  mc2nu_data = GData(mc2nu_file)
  cdim = mc2nu_data.get_num_dims()

  _, mc2nu_values = GInterpModal(mc2nu_data, 1, "ms").interpolate(tuple(range(cdim)))

  nonuniform_grid = list(uniform_grid)
  for d in range(cdim):
    mc2nu_single_axis = _extract_values_along_dimension(mc2nu_values, d, cdim)
    nonuniform_grid[d] = _convert_cell_centered_to_nodal(mc2nu_single_axis)
  # end
  return nonuniform_grid
# end

# Public API
def load_gk_distf(
    name: str, species: str, frame: int,
    tag: str = "f", source: bool = False, use_c2p_vel: bool = False,
    use_mc2nu: bool = False, block_idx: int | None = None,
) -> GData:
  """Build a real distribution function from saved JBf data."""
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = "source_" if source else ""

  jf_file           = f"{prefix}-{species}_{frame_infix}{frame}.gkyl"
  mapc2p_vel_file   = f"{prefix}-{species}_mapc2p_vel.gkyl"
  jacobvel_file     = f"{prefix}-{species}_jacobvel.gkyl"
  mc2nu_file        = f"{prefix}-mc2nu_pos_deflated.gkyl"
  jacobtot_inv_file = f"{prefix}-jacobtot_inv.gkyl"

  jf_data           = GData(jf_file, mapc2p_vel_name=mapc2p_vel_file if use_c2p_vel else None)
  jacobvel_data     = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  # Divide Jf by jacobvel to get f * J_x * B.
  fjxB_data = GData(ctx=jf_data.ctx) # Inside a GData object so we can interpolate
  fjxB_values = jf_data.get_values() / jacobvel_data.get_values()
  fjxB_data.push(jf_data.get_grid(), fjxB_values)

  # Interpolate f * J_x * B and jacobtot_inv to the same grid.
  out_grid, fjxB_values    = GInterpModal(fjxB_data, 1, "gkhyb").interpolate()
  _, jacobtot_inv_values   = GInterpModal(jacobtot_inv_data, 1, "ms").interpolate()
  fjxB_values              = np.squeeze(fjxB_values)
  jacobtot_inv_values      = np.squeeze(jacobtot_inv_values)

  # Reshape jacobtot_inv to have 1 component over velocity dimensions, then multiply.
  vdim = fjxB_values.ndim - jacobtot_inv_values.ndim
  jacobtot_inv_reshaped = jacobtot_inv_values.reshape(jacobtot_inv_values.shape + (1,) * vdim)
  f_values = fjxB_values * jacobtot_inv_reshaped
  # Add 1 dimension to represent 1 component
  f_values = f_values.reshape(f_values.shape + (1,))

  if use_mc2nu:
    out_grid = _apply_mc2nu_grid(out_grid, mc2nu_file)
    if use_c2p_vel:
      jf_data.ctx["grid_type"] = "c2p_vel + mc2nu"
    else:
      jf_data.ctx["grid_type"] = "mc2nu"
    # end
  # end

  out = GData(tag=tag, ctx=jf_data.ctx)
  out.push(out_grid, f_values)
  return out
# end

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
@click.option("--mc2nu", "-m", is_flag=True, default=False,
    help="Use <name>-mc2nu_pos_deflated.gkyl to deform the configuration-space grid.")
@click.option("--block", "-b", default=None, type=click.INT,
  help="Use block-specific files with _b<idx> prefix, e.g. -b 1 loads <name>_b1-*.gkyl.")
@click.option("--tag", "-t", default="f", type=click.STRING,
    help="Tag for output dataset.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  data = ctx.obj["data"]

  verb_print(ctx, "Building distribution function for " + kwargs["name"])

  frame_spec = kwargs["frame"].strip()
  if "," in frame_spec:
    frames = [int(f.strip()) for f in frame_spec.split(",")] # List of frames specified on input
  elif ":" not in frame_spec:
    frames = [int(frame_spec)] # Stick to the frame specified on input
  else:
    # Figure out how many frames are possible to read based on what files are available
    prefix = f"{kwargs['name']}_b{kwargs['block']}" if kwargs["block"] is not None else kwargs["name"]
    frame_infix = "source_" if kwargs["source"] else ""
    stem = f"{prefix}-{kwargs['species']}_{frame_infix}"
    available = sorted({
        int(f.removeprefix(stem)[:-5])
        for f in glob.glob(f"{glob.escape(stem)}*.gkyl")
        if f.removeprefix(stem)[:-5].isdigit()
    })
    # Slice the data accordingly
    parts = frame_spec.split(":")
    lower = int(parts[0]) if parts[0] else available[0]
    upper = int(parts[1]) if parts[1] else available[-1] + 1
    step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    frames = [f for f in available if lower <= f < upper and (f - lower) % step == 0]
  # end
  verb_print(ctx, f"Loading frames: {frames}")

  for frame in frames:
    out = load_gk_distf(
        name=kwargs["name"], species=kwargs["species"], frame=frame,
        tag=kwargs["tag"], source=kwargs["source"],
        use_c2p_vel=kwargs["c2p_vel"], use_mc2nu=kwargs["mc2nu"],
        block_idx=kwargs["block"],
    )
    data.add(out)
  # end

  if len(frames) > 1:
    data.set_unique_labels()
  # end
# end