import glob
import os

import click
import numpy as np

from postgkyl.data import GData
from postgkyl.data import GInterpModal
from postgkyl.utils import verb_print
import postgkyl.utils.gk_utils as gku


def _resolve_path(path: str, file_name: str) -> str:
  if os.path.isabs(file_name):
    return file_name
  return os.path.join(path, file_name)


def _discover_block_indices(name: str, species: str, path: str,
    source: bool = False) -> list[int]:
  """Return sorted unique block indices by parsing filenames on disk.

  Matches ``{name}_b{N}-{species}_*.gkyl`` and extracts N from each filename,
  so the result is independent of how many frame files exist per block.
  """
  src_infix = "source_" if source else ""
  template = _resolve_path(path, f"{name}_b*-{species}_{src_infix}*.gkyl")
  files = glob.glob(template)
  prefix = f"{name}_b"
  sep = f"-{species}_"
  indices = set()
  for f in files:
    base = os.path.basename(f)
    if base.startswith(prefix) and sep in base:
      block_str = base[len(prefix):base.index(sep)]
      try:
        indices.add(int(block_str))
      except ValueError:
        pass
  return sorted(indices)


def _resolve_files(name: str, species: str, frame: int, path: str,
    source: bool = False, block_idx: int | None = None):
  prefix = f"{name}_b{block_idx}" if block_idx is not None else name
  if source:
    frame_file = _resolve_path(path, f"{prefix}-{species}_source_{frame}.gkyl")
  else:
    frame_file = _resolve_path(path, f"{prefix}-{species}_{frame}.gkyl")
  mapc2p_vel_file = _resolve_path(path, f"{prefix}-{species}_mapc2p_vel.gkyl")
  mc2nu_file = _resolve_path(path, f"{prefix}-mc2nu_pos_deflated.gkyl")
  jacobvel_file = _resolve_path(path, f"{prefix}-{species}_jacobvel.gkyl")
  jacobtot_inv_file = _resolve_path(path, f"{prefix}-jacobtot_inv.gkyl")
  return frame_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file


def _resolve_available_frames(name: str, species: str, path: str,
    source: bool = False, block_idx: int | None = None) -> list[int]:
  prefix_name = f"{name}_b{block_idx}" if block_idx is not None else name
  if source:
    prefix = _resolve_path(path, f"{prefix_name}-{species}_source_")
  else:
    prefix = _resolve_path(path, f"{prefix_name}-{species}_")

  files = glob.glob(f"{glob.escape(prefix)}*.gkyl")
  frames = []
  for file_name in files:
    suffix = file_name.removeprefix(prefix)
    if suffix.endswith(".gkyl"):
      frame_str = suffix[:-5]
      try:
        frames.append(int(frame_str))
      except ValueError:
        pass

  return sorted(set(frames))


def _parse_frame_bounds(frame: str) -> tuple[int | None, int | None, int]:
  parts = frame.split(":")
  if len(parts) < 2 or len(parts) > 3:
    raise ValueError(
        "Frame range must use the form 'start:stop[:step]' or ':' for all frames."
    )

  try:
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
  except ValueError as err:
    raise ValueError(
        "Frame range must use integer bounds, e.g. ':', '40:50', or '40:50:2'."
    ) from err

  if step <= 0:
    raise ValueError("Frame range step must be a positive integer.")

  return start, stop, step


def _resolve_frames(name: str, species: str, frame: str, path: str,
    source: bool = False, block_idx: int | None = None) -> list[int]:
  frame = frame.strip()
  if not frame:
    raise ValueError("Frame cannot be empty.")

  if ":" not in frame:
    try:
      return [int(frame)]
    except ValueError as err:
      raise ValueError(
          "Frame must be an integer or a slice like ':', '40:50', or '40:50:2'."
      ) from err

  available_frames = _resolve_available_frames(name, species, path, source,
      block_idx=block_idx)
  if not available_frames:
    raise FileNotFoundError(
        f"No frame files found for {name}-{species} in {path}."
    )

  start, stop, step = _parse_frame_bounds(frame)
  lower = available_frames[0] if start is None else start
  upper = available_frames[-1] + 1 if stop is None else stop

  frames = [
      frame_num for frame_num in available_frames
      if lower <= frame_num < upper and (frame_num - lower) % step == 0
  ]
  if not frames:
    raise FileNotFoundError(
        f"No frames matched '{frame}' for {name}-{species} in {path}."
    )

  return frames


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


def _find_stitch_axis(grids: list) -> int:
  """Return the axis along which consecutive blocks are non-overlapping.

  The stitch axis is the dimension where every consecutive block pair starts
  no earlier than the previous block ends. Velocity-space axes always share
  the same range across blocks so they never qualify.
  """
  ndim = len(grids[0])
  for axis in range(ndim):
    is_stitch = True
    for b in range(len(grids) - 1):
      max_curr = float(np.asarray(grids[b][axis]).max())
      min_next = float(np.asarray(grids[b + 1][axis]).min())
      tol = 1e-10 * max(abs(max_curr), 1.0)
      if min_next < max_curr - tol:
        is_stitch = False
        break
    if is_stitch:
      return axis
  raise ValueError(
      "Could not determine stitch axis: no single axis is non-overlapping "
      "across all consecutive block pairs."
  )


def _stitch_block_data(block_outputs: list, tag: str) -> GData:
  """Concatenate per-block GData objects into one dataset along the stitch axis."""
  grids = [out.get_grid() for out in block_outputs]
  stitch_axis = _find_stitch_axis(grids)

  # Sort blocks by their minimum coordinate on the stitch axis.
  order = sorted(range(len(block_outputs)),
                 key=lambda i: float(np.asarray(grids[i][stitch_axis]).min()))
  block_outputs = [block_outputs[i] for i in order]
  grids = [block_outputs[i].get_grid() for i in range(len(block_outputs))]
  values = [block_outputs[i].get_values() for i in range(len(block_outputs))]

  # Build merged grid.  Node coordinates share a boundary, so drop the
  # first node of each subsequent block along the stitch axis.
  ndim = len(grids[0])
  merged_grid = []
  for d in range(ndim):
    if d == stitch_axis:
      merged_axis = np.concatenate(
          [np.asarray(grids[0][d])] +
          [np.asarray(grids[b][d][1:]) for b in range(1, len(grids))]
      )
    else:
      merged_axis = np.asarray(grids[0][d])
    merged_grid.append(merged_axis)

  # Concatenate values along the spatial stitch axis.
  merged_values = np.concatenate(values, axis=stitch_axis)

  result = GData(tag=tag, ctx=block_outputs[0].ctx)
  result.push(merged_grid, merged_values)
  return result


def _interpolate_fjx_and_jacob(fjx_data: GData, jacobtot_inv_data: GData):
  fjx_interp = GInterpModal(fjx_data, 1, "gkhyb")
  jacob_interp = GInterpModal(jacobtot_inv_data, 1, "ms")

  grid, fjx_values = fjx_interp.interpolate()
  _, jacob_values = jacob_interp.interpolate()

  fjx_values = np.squeeze(fjx_values)
  jacob_values = np.squeeze(jacob_values)
  return grid, fjx_values, jacob_values


def load_gk_distf(name: str, species: str, frame: int, path: str = "./",
    tag: str = "df", source: bool = False, use_c2p_vel: bool = False,
    use_mc2nu: bool = False, debug: bool = False,
    block_idx: int | None = None) -> GData:
  """Build a real distribution function from saved Jf data.

  This is the Python API used by the ``gk-distf`` CLI command.
  """
  files = {}
  jf_file, mapc2p_vel_file, mc2nu_file, jacobvel_file, jacobtot_inv_file = _resolve_files(
      name, species, frame, path, source, block_idx=block_idx)
  files["Jf"] = jf_file
  files["jacobvel"] = jacobvel_file
  files["jacobtot_inv"] = jacobtot_inv_file

  if use_c2p_vel:
    files["mapc2p_vel"] = mapc2p_vel_file
  if use_mc2nu:
    files["mc2nu"] = mc2nu_file

  _assert_files_exist(files)

  map_kwargs = _resolve_mapping_kwargs(use_c2p_vel, mapc2p_vel_file)

  if debug:
    click.echo(f"gk_distf: Jf={jf_file}")
    click.echo(f"gk_distf: jacobvel={jacobvel_file}")
    click.echo(f"gk_distf: jacobtot_inv={jacobtot_inv_file}")
    if use_c2p_vel:
      click.echo(f"gk_distf: mapc2p_vel={mapc2p_vel_file}")
    if use_mc2nu:
      click.echo(f"gk_distf: mc2nu={mc2nu_file}")

  jf_data = GData(jf_file, mapc2p_vel_name=map_kwargs["mapc2p_vel_name"])
  jacobvel_data = GData(jacobvel_file)
  jacobtot_inv_data = GData(jacobtot_inv_file)

  fjx_data = GData(tag=tag, ctx=jf_data.ctx)
  fjx_values = _compute_fjx(jf_data.get_values(), jacobvel_data.get_values())
  fjx_data.push(jf_data.get_grid(), fjx_values)

  out_grid, fjx_interp, jacob_interp = _interpolate_fjx_and_jacob(fjx_data, jacobtot_inv_data)
  distf_values = _broadcast_multiply(fjx_interp, jacob_interp)

  if use_mc2nu:
    out_grid, _, _ = _apply_mc2nu_grid(out_grid, mc2nu_file, debug)
    jf_data.ctx["grid_type"] = "mc2nu"

  if debug:
    click.echo(f"gk_distf: output shape={distf_values.shape}")

  out = GData(tag=tag, ctx=jf_data.ctx)
  out.push(out_grid, np.asarray(distf_values)[..., np.newaxis])
  return out


def load_gk_distf_mb(name: str, species: str, frame: int, path: str = "./",
    tag: str = "df", source: bool = False, use_c2p_vel: bool = False,
    use_mc2nu: bool = False, debug: bool = False,
    mb_blocks: str = "-1") -> GData:
  """Load and stitch a multiblock distribution function.

  Files are expected as ``{name}_b{N}-{species}_{frame}.gkyl`` with the
  same ``_b{N}`` prefix on all auxiliary files (jacobvel, jacobtot_inv,
  mapc2p_vel, mc2nu).
  """
  if mb_blocks == "-1":
    blocks = _discover_block_indices(name, species, path, source)
  else:
    blocks = gku.get_block_indices(mb_blocks, "")

  if not blocks:
    raise FileNotFoundError(
        f"No block files found for {name}_b*-{species} in {path}")

  if debug:
    click.echo(f"gk_distf: multiblock mode, blocks={blocks}")

  block_outputs = []
  for block_idx in blocks:
    out = load_gk_distf(
        name=name, species=species, frame=frame, path=path,
        tag=f"{tag}_b{block_idx}", source=source,
        use_c2p_vel=use_c2p_vel, use_mc2nu=use_mc2nu,
        debug=debug, block_idx=block_idx)
    block_outputs.append(out)

  return _stitch_block_data(block_outputs, tag)


@click.command()
@click.option("--name", "-n", required=True, type=click.STRING,
    help="Simulation name prefix (e.g. gk_lorentzian_mirror).")
@click.option("--species", "-s", required=True, type=click.STRING,
    help="Species name (e.g. ion or elc).")
@click.option("--frame", "-f", required=True, type=click.STRING,
  help="Frame number or range. Use ':' for all frames and 'start:stop[:step]' for ranges.")
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
@click.option("--mb", "-b", is_flag=True, default=False,
    help="Multiblock mode. Files are expected as {name}_b{N}-{species}_{frame}.gkyl.")
@click.option("--mb-blocks", type=click.STRING, default=None,
    help=("Restrict multiblock to specific blocks: comma-separated list or slice "
          "(start:stop:step). Requires --mb. Defaults to all discovered blocks."))
@click.option("--debug", is_flag=True,
    help="Print resolved file names and shape diagnostics.")
@click.pass_context
def gk_distf(ctx, **kwargs):
  """Gyrokinetics: build real distribution function from saved Jf data."""
  verb_print(ctx, "Starting gk_distf")
  data = ctx.obj["data"]

  is_multib = kwargs["mb"]
  mb_blocks_str = kwargs["mb_blocks"] if kwargs["mb_blocks"] is not None else "-1"

  # For frame-range resolution, use block 0 as the reference when multiblock.
  ref_block = None
  if is_multib:
    if mb_blocks_str == "-1":
      blocks = _discover_block_indices(
          kwargs["name"], kwargs["species"], kwargs["path"], kwargs["source"])
    else:
      blocks = gku.get_block_indices(mb_blocks_str, "")
    if not blocks:
      ctx.fail(click.style(
          f"No block files found for {kwargs['name']} in {kwargs['path']}", fg="red"))
    ref_block = blocks[0]

  try:
    frames = _resolve_frames(
        name=kwargs["name"],
        species=kwargs["species"],
        frame=kwargs["frame"],
        path=kwargs["path"],
        source=kwargs["source"],
        block_idx=ref_block)
  except (FileNotFoundError, ValueError) as err:
    ctx.fail(click.style(str(err), fg="red"))

  if kwargs["debug"] and len(frames) > 1:
    click.echo(f"gk_distf: resolved frames={frames}")

  for frame in frames:
    if is_multib:
      out = load_gk_distf_mb(
          name=kwargs["name"],
          species=kwargs["species"],
          frame=frame,
          path=kwargs["path"],
          tag=kwargs["tag"],
          source=kwargs["source"],
          use_c2p_vel=kwargs["c2p_vel"],
          use_mc2nu=kwargs["mc2nu"],
          debug=kwargs["debug"],
          mb_blocks=mb_blocks_str)
    else:
      out = load_gk_distf(
          name=kwargs["name"],
          species=kwargs["species"],
          frame=frame,
          path=kwargs["path"],
          tag=kwargs["tag"],
          source=kwargs["source"],
          use_c2p_vel=kwargs["c2p_vel"],
          use_mc2nu=kwargs["mc2nu"],
          debug=kwargs["debug"])
    data.add(out)

  if len(frames) > 1:
    data.set_unique_labels()

  verb_print(ctx, "Finishing gk_distf")