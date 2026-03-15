import glob
import os

import click
import numpy as np

from postgkyl.data import GData, GInterpModal
from postgkyl.utils import verb_print


# ---------------------------------------------------------------------------
# File/path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str, file_name: str) -> str:
  if os.path.isabs(file_name):
    return file_name
  # end
  return os.path.join(path, file_name)
# end


def _resolve_files(
    name: str, species: str, frame: int, path: str,
    source: bool = False, block_idx: int | None = None,
) -> tuple[str, str, str, str, str]:
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = "source_" if source else ""
  frame_file      = _resolve_path(path, f"{prefix}-{species}_{frame_infix}{frame}.gkyl")
  mapc2p_vel_file = _resolve_path(path, f"{prefix}-{species}_mapc2p_vel.gkyl")
  mc2nu_file      = _resolve_path(path, f"{prefix}-mc2nu_pos_deflated.gkyl")
  jacobvel_file   = _resolve_path(path, f"{prefix}-{species}_jacobvel.gkyl")
  jacobtot_inv_file = _resolve_path(path, f"{prefix}-jacobtot_inv.gkyl")
  return frame_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file
# end


def _error_check(check: str, **kwargs):
  """Centralized validation for gk_distf helpers.

  This keeps error checks in one place while helper functions focus on data flow.
  """
  if check == "frame_nonempty":
    if not kwargs["frame"]:
      raise ValueError("Frame cannot be empty.")
    # end
  # end

  if check == "single_frame":
    try:
      int(kwargs["frame"])
    except ValueError as err:
      raise ValueError(
          "Frame must be an integer or a slice like ':', '40:50', or '40:50:2'."
      ) from err
    # end
  # end

  if check == "frame_bounds":
    parts = kwargs["frame"].split(":")
    if len(parts) < 2 or len(parts) > 3:
      raise ValueError(
          "Frame range must use the form 'start:stop[:step]' or ':' for all frames."
      )
    # end

    try:
      int(parts[0]) if parts[0] else None
      int(parts[1]) if parts[1] else None
      step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    except ValueError as err:
      raise ValueError(
          "Frame range must use integer bounds, e.g. ':', '40:50', or '40:50:2'."
      ) from err
    # end

    if step <= 0:
      raise ValueError("Frame range step must be a positive integer.")
    # end
  # end

  if check == "available_frames":
    if not kwargs["available"]:
      raise FileNotFoundError(
          f"No frame files found for {kwargs['name']}-{kwargs['species']} in {kwargs['path']}."
      )
    # end
  # end

  if check == "matched_frames":
    if not kwargs["frames"]:
      raise FileNotFoundError(
          f"No frames matched '{kwargs['frame']}' for {kwargs['name']}-{kwargs['species']} in {kwargs['path']}."
      )
    # end
  # end

  if check == "required_files":
    missing = [
        f"{label}: {file_name}"
        for label, file_name in kwargs["files"].items()
        if not os.path.exists(file_name)
    ]
    if missing:
      raise FileNotFoundError("Missing required files:\n  - " + "\n  - ".join(missing))
    # end
  # end

  if check == "fjx_shape":
    jf_shape = kwargs["jf_shape"]
    jac_shape = kwargs["jac_shape"]
    if jf_shape != jac_shape:
      if jf_shape[:-1] != jac_shape[:-1]:
        raise ValueError(
            "Jf and jacobvel spatial shapes do not match: "
            f"{jf_shape} vs {jac_shape}"
        )
      # end
      if jac_shape[-1] != 1:
        raise ValueError(
            "jacobvel component axis must be 1 or match Jf: "
            f"{jf_shape} vs {jac_shape}"
        )
      # end
    # end
  # end

  if check == "broadcast_shape":
    distf_shape = kwargs["distf_shape"]
    jac_shape = kwargs["jac_shape"]
    if len(jac_shape) > len(distf_shape):
      raise ValueError(
          "jacobtot_inv has more dimensions than distribution function: "
          f"{len(jac_shape)} > {len(distf_shape)}"
      )
    # end

    if distf_shape[:len(jac_shape)] != jac_shape:
      raise ValueError(
          "Cannot broadcast jacobtot_inv over distribution function. "
          f"Expected prefix {distf_shape[:len(jac_shape)]}, got {jac_shape}"
      )
    # end
  # end

  if check == "mc2nu_cdim":
    cdim = kwargs["cdim"]
    total_dims = kwargs["total_dims"]
    if cdim < 1 or cdim > total_dims:
      raise ValueError(f"Invalid cdim={cdim} for total_dims={total_dims}")
    # end
  # end

  if check == "mc2nu_axis":
    mapped_size = kwargs["mapped_size"]
    old_size = kwargs["old_size"]
    axis = kwargs["axis"]
    if mapped_size != old_size and mapped_size + 1 != old_size:
      raise ValueError(
          f"mc2nu mapping size is incompatible with configuration grid on axis {axis}: "
          f"{mapped_size} vs {old_size}"
      )
    # end
  # end

  raise ValueError(f"Unknown precheck type '{check}'")
# end


# ---------------------------------------------------------------------------
# Frame resolution helpers
# ---------------------------------------------------------------------------

def _resolve_available_frames(
    name: str, species: str, path: str,
    source: bool = False, block_idx: int | None = None,) -> list[int]:
  """Scan the path for available frame files matching the naming pattern."""
  prefix_name = f"{name}_b{block_idx}" if block_idx is not None else name
  frame_infix = "source_" if source else ""
  prefix = _resolve_path(path, f"{prefix_name}-{species}_{frame_infix}")

  frames = []
  for file_name in glob.glob(f"{glob.escape(prefix)}*.gkyl"):
    suffix = file_name.removeprefix(prefix)
    if suffix.endswith(".gkyl"):
      try:
        frames.append(int(suffix[:-5]))
      except ValueError:
        pass
      # end
    # end
  # end
  return sorted(set(frames))
# end


def _resolve_frames(
    name: str, species: str, frame: str, path: str,
    source: bool = False, block_idx: int | None = None,
) -> list[int]:
  """Parse frame input and resolve to a list of available frame numbers."""
  frame = frame.strip()
  _error_check("frame_nonempty", frame=frame)

  if ":" not in frame:
    _error_check("single_frame", frame=frame)
    return [int(frame)]
  # end

  available = _resolve_available_frames(name, species, path, source, block_idx=block_idx)
  _error_check("available_frames", available=available, name=name, species=species, path=path)
  _error_check("frame_bounds", frame=frame)

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
  _error_check("matched_frames", frames=frames, frame=frame, name=name, species=species, path=path)
  return frames
# end

# ---------------------------------------------------------------------------
# Physics / DG helpers
# ---------------------------------------------------------------------------

def _compute_fjx(jf_values: np.ndarray, jac_values: np.ndarray) -> np.ndarray:
  """Divide Jf by jacobvel to get f * J_x * B. Could be improved with weak division"""
  _error_check("fjx_shape", jf_shape=jf_values.shape, jac_shape=jac_values.shape)
  return jf_values / jac_values
# end


def _broadcast_multiply(distf: np.ndarray, jacobtot_inv: np.ndarray) -> np.ndarray:
  """Multiply f * J_x * B by jacobtot_inv, broadcasting over the component axis."""
  _error_check("broadcast_shape", distf_shape=distf.shape, jac_shape=jacobtot_inv.shape)
  jacob_shape = jacobtot_inv.shape + (1,) * (distf.ndim - jacobtot_inv.ndim)
  return distf * jacobtot_inv.reshape(jacob_shape)
# end


def _interpolate_distf_components(
    fjx_data: GData, jacobtot_inv_data: GData,
) -> tuple[list, np.ndarray, np.ndarray]:
  """Interpolate f * J_x * B (gkhyb basis) and jacobtot_inv (ms basis)."""
  grid, fjx_values = GInterpModal(fjx_data, 1, "gkhyb").interpolate()
  _, jacob_values = GInterpModal(jacobtot_inv_data, 1, "ms").interpolate()
  return grid, np.squeeze(fjx_values), np.squeeze(jacob_values)
# end

# ---------------------------------------------------------------------------
# mc2nu grid deformation helpers
# ---------------------------------------------------------------------------

def _cell_centers_to_nodes(cell_centers: np.ndarray) -> np.ndarray:
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
  """Replace configuration-space axes in out_grid with mc2nu-mapped coordinates."""
  mc2nu_data = GData(mc2nu_file)
  cdim = mc2nu_data.get_num_dims()
  _error_check("mc2nu_cdim", cdim=cdim, total_dims=len(out_grid))

  mc2nu_interp = GInterpModal(mc2nu_data, 1, "ms")
  _, mc2nu_values = mc2nu_interp.interpolate(tuple(range(cdim)))
  mapped_values = np.asarray(mc2nu_values)

  deformed_grid = list(out_grid)
  for d in range(cdim):
    mapped_cfg = _extract_mapped_axis(mapped_values, d, cdim)
    old_cfg = np.asarray(out_grid[d])
    _error_check("mc2nu_axis", mapped_size=mapped_cfg.size, old_size=old_cfg.size, axis=d)
    if mapped_cfg.size == old_cfg.size:
      deformed_grid[d] = mapped_cfg
    else:
      deformed_grid[d] = _cell_centers_to_nodes(mapped_cfg)
    # end
  # end

  verb_print(ctx, f"gk_distf: mc2nu mapped {cdim} configuration axis/axes")
  return deformed_grid
# end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gk_distf(
    name: str, species: str, frame: int, path: str = "./",
    tag: str = "df", source: bool = False, use_c2p_vel: bool = False,
    use_mc2nu: bool = False,
    block_idx: int | None = None,
    ctx=None,
) -> GData:
  """Build a real distribution function from saved JBf data."""
  jf_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file = _resolve_files(
      name, species, frame, path, source, block_idx=block_idx)

  required_files = {"Jf": jf_file, "jacobvel": jacobvel_file, "jacobtot_inv": jacobtot_inv_file}
  if use_c2p_vel:
    required_files["mapc2p_vel"] = mapc2p_vel_file
  # end
  if use_mc2nu:
    required_files["mc2nu"] = mc2nu_file
  # end
  _error_check("required_files", files=required_files)

  for kind, fname in required_files.items():
    verb_print(ctx, f"gk_distf: {kind}={fname}")
  # end

  jf_data = GData(jf_file, mapc2p_vel_name=mapc2p_vel_file if use_c2p_vel else "")
  jacobvel_data = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  fjxB_data = GData(tag=tag, ctx=jf_data.ctx)
  fjxB_data.push(jf_data.get_grid(), _compute_fjx(jf_data.get_values(), jacobvel_data.get_values()))

  out_grid, fjx_values, jacob_values = _interpolate_distf_components(fjxB_data, jacobtot_inv_data)
  distf_values = _broadcast_multiply(fjx_values, jacob_values)

  if use_mc2nu:
    out_grid = _apply_mc2nu_grid(out_grid, mc2nu_file, ctx)
    jf_data.ctx["grid_type"] = "mc2nu"
  # end

  verb_print(ctx, f"gk_distf: output shape={distf_values.shape}")

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
@click.option("--frame", "-f", required=True, type=click.STRING,
    help="Frame number or range. Use ':' for all frames and 'start:stop[:step]' for ranges.")
@click.option("--path", "-p", default="./", type=click.STRING,
    help="Path to simulation data.")
@click.option("--tag", "-t", default="df", type=click.STRING,
    help="Tag for output dataset.")
@click.option("--source", is_flag=True,
    help="Use <name>-<species>_source_<frame>.gkyl as the input distribution.")
@click.option("--c2p-vel", "-v", is_flag=True, default=False,
    help="Use <name>-<species>_mapc2p_vel.gkyl when loading Jf.")
@click.option("--mc2nu", "-m", is_flag=True,
    help="Use <name>-mc2nu_pos_deflated.gkyl to deform the configuration-space grid.")
@click.option("--block", "-b", default=None, type=click.INT,
  help="Use block-specific files with _b<idx> prefix, e.g. -b 1 loads <name>_b1-*.gkyl.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  verb_print(ctx, "Starting gk_distf")
  data = ctx.obj["data"]

  try:
    frames = _resolve_frames(
        name=kwargs["name"], species=kwargs["species"],
        frame=kwargs["frame"], path=kwargs["path"],
        source=kwargs["source"], block_idx=kwargs["block"])
  except (FileNotFoundError, ValueError) as err:
    ctx.fail(click.style(str(err), fg="red"))
  # end

  if len(frames) > 1:
    verb_print(ctx, f"gk_distf: resolved frames={frames}")
  # end

  common = dict(
      name=kwargs["name"], species=kwargs["species"],
      path=kwargs["path"], tag=kwargs["tag"],
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

  verb_print(ctx, "Finishing gk_distf")
# end