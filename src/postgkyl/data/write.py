"""Write helpers for GData."""

from typing import Literal
import json
import os
import re
import shutil

import numpy as np

try:
  import adios2
  has_adios = True
except ModuleNotFoundError:
  has_adios = False
# end


def write(self, out_name: str = "",
    extension: Literal["gkyl", "bp", "txt", "npy", "vts"] = "gkyl",
    mode: str = "", var_name: str = "", append: bool = False,
    cleaning: bool = True, norm_axes: bool = False) -> None:
  """Writes data in a file.

  The available formats are Gkeyll .gkyl (default), ADIOS .bp file, ASCII .txt file,
  NumPy .npy file, or VTK structured grid .vts file.

  Args:
    out_name: str
      Specify output file name.
    extension: str = "gkyl"
      Specify file extension (extension).
    var_name: str
      Specify variable name for Adios.
    append: bool = False
      Allows for writing multiple datasets into one file.
    cleaning: bool = True
      Remove temporary files after writing.
    norm_axes: bool = False
      Normalize axes to [-1, 1] for VTK output.

  Returns:
    None
  """

  if mode:
    extension = mode
    print("Deprecation warning: mode of the write method is going to be renamed to extension.")
  # end

  if not out_name:
    if self._file_name is not None:
      fn = self._file_name
      out_name = f"{fn.split('.', maxsplit=1)[0].strip('_')}_mod.{extension}"
    else:
      out_name = f"gdata.{extension}"
    # end
  else:
    if not isinstance(out_name, str):
      raise TypeError("'out_name' must be a string")
    # end
    if out_name.split(".")[-1] != extension:
      out_name += "." + extension
    # end
  # end

  num_dims = self.num_dims
  num_comps = self.num_comps
  num_cells = self.num_cells
  lo, up = self.bounds
  values = self.values

  full_shape = list(num_cells) + [num_comps]
  offset = [0] * (num_dims + 1)

  if not var_name:
    var_name = self._var_name
  # end

  if extension == "bp":
    if not has_adios:
      raise ModuleNotFoundError("ADIOS2 is not installed")
    # end

    if not append:
      fh = adios2.open(out_name, "w", engine_type="BP3")
      fh.write_attribute("numCells", num_cells)
      fh.write_attribute("lowerBounds", lo)
      fh.write_attribute("upperBounds", up)

      if self.ctx["time"]:
        fh.write("time", self.ctx["time"])
      # end
    else:
      fh = adios2.open(out_name, "a", engine_type="BP3")
    # end
    fh.write(var_name, values, full_shape, offset, full_shape)
    fh.close()

    if cleaning:
      if len(out_name.split("/")) > 1:
        nm = out_name.split("/")[-1]
      else:
        nm = out_name
      # end
      shutil.move(f"{out_name}.dir/{nm}.0", f"{out_name}")
      shutil.rmtree(f"{out_name}.dir")
    # end
  elif extension == "gkyl":
    dti = np.dtype("i8")
    dtf = np.dtype("f8")

    fh = open(out_name, "w", encoding="utf-8")

    # sep='' results in a binary file
    np.array([103, 107, 121, 108, 48], dtype=np.dtype("b")).tofile(fh, sep="")
    # version 1
    np.array([1], dtype=dti).tofile(fh, sep="")
    # type 1
    np.array([1], dtype=dti).tofile(fh, sep="")
    # meta size
    np.array([0], dtype=dti).tofile(fh, sep="")
    # real type (double)
    np.array([2], dtype=dti).tofile(fh, sep="")
    # num dims
    np.array([num_dims], dtype=dti).tofile(fh, sep="")
    # num cells
    np.array(num_cells, dtype=dti).tofile(fh, sep="")
    # lower
    np.array(lo, dtype=dtf).tofile(fh, sep="")
    # upper
    np.array(up, dtype=dtf).tofile(fh, sep="")
    # elem_sz
    np.array([num_comps * 8], dtype=dti).tofile(fh, sep="")
    # asize
    np.array([np.size(values)], dtype=dti).tofile(fh, sep="")
    # data
    np.array(values, dtype=dtf).tofile(fh, sep="")

    fh.close()
  elif extension == "txt":
    num_rows = np.prod(num_cells)
    grid = self.get_grid()
    for d in range(num_dims):
      grid[d] = 0.5 * (grid[d][1:] + grid[d][:-1])
    # end

    basis = np.full(num_dims, 1.0)
    for d in range(num_dims - 1):
      basis[d] = np.prod(num_cells[(d + 1) :])
    # end

    fh = open(out_name, "w", encoding="utf-8")
    for i in range(num_rows):
      idx = i
      idxs = np.zeros(num_dims, np.int32)
      for d in range(num_dims):
        idxs[d] = int(idx // basis[d])
        idx = idx % basis[d]
      # end
      line = ""
      for d in range(num_dims):
        line += f"{grid[d][idxs[d]]:.15e}, "
      # end
      for c in range(num_comps - 1):
        line += f"{values[tuple(idxs)][c]:.15e}, "
      # end
      line += f"{values[tuple(idxs)][num_comps - 1]:.15e}\n"
      fh.write(line)
    # end
    fh.close()
  elif extension == "npy":
    np.save(out_name, values.squeeze())
  # end
  elif extension == "vts":
    # To plot Gkeyll data in virtual reality (VR). Maxwell Rosen reccomends
    # Outputtng data in .vts format and importing it into Paraview, which has a VR interface.
    import pyvista as pv
    from postgkyl.output.nodal_to_cell_centered_grid import nodal_to_cell_centered_grid

    n_grid = nodal_to_cell_centered_grid(self.get_grid(), num_cells, meshgrid=True)
    if num_dims == 1:
      fval = values.squeeze()
      X = n_grid[0]
      Y = np.zeros_like(X)
      Z = fval
    elif num_dims == 2:
      fval = values.squeeze()
      X, Y = n_grid
      Z = fval
    elif num_dims == 3:
      fval = values.squeeze()
      X, Y, Z = n_grid

    if norm_axes:  # Normalize to [-1, 1]
      X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
      Y = 2 * (Y - Y.min()) / (Y.max() - Y.min()) - 1
      Z = 2 * (Z - Z.min()) / (Z.max() - Z.min()) - 1

    grid3d = pv.StructuredGrid(X, Y, Z)
    grid3d["f_raw"] = fval.ravel(order="F")
    grid3d.save(out_name)
    _update_vtk_series_file(self, out_name)


def _update_vtk_series_file(self, out_name: str) -> None:
  """Create or update ParaView .series metadata for VTK file-series time playback."""
  out_dir = os.path.dirname(out_name)
  out_file = os.path.basename(out_name)
  stem, ext = os.path.splitext(out_file)
  match = re.match(r"^(.*?)(?:[_-]?(\d+))$", stem)
  if match and match.group(1):
    series_stem = match.group(1).rstrip("_-")
    if not series_stem:
      series_stem = stem
  else:
    series_stem = stem
  # end

  series_path = os.path.join(out_dir, f"{series_stem}{ext}.series")
  time_value = float(self.ctx.get("time", self.ctx.get("frame", 0.0)))
  rel_file = os.path.relpath(out_name, out_dir if out_dir else ".")

  series_data = {"file-series-version": "1.0", "files": []}
  if os.path.exists(series_path):
    try:
      with open(series_path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
      if isinstance(loaded, dict) and isinstance(loaded.get("files"), list):
        series_data = loaded
        if "file-series-version" not in series_data:
          series_data["file-series-version"] = "1.0"
        # end
    except (OSError, json.JSONDecodeError):
      pass
    # end
  # end

  replaced = False
  for entry in series_data["files"]:
    if entry.get("name") == rel_file:
      entry["time"] = time_value
      replaced = True
      break
    # end
  # end
  if not replaced:
    series_data["files"].append({"name": rel_file, "time": time_value})
  # end

  series_data["files"].sort(key=lambda x: (float(x.get("time", 0.0)), x.get("name", "")))
  with open(series_path, "w", encoding="utf-8") as fh:
    json.dump(series_data, fh, indent=2)
    fh.write("\n")
