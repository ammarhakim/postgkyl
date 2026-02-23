"""Module including Gkeyll data class"""

from typing import Literal, Tuple
import numpy as np
import shutil

try:
  import adios2
  has_adios = True
except ModuleNotFoundError:
  has_adios = False
# end

from postgkyl.data.gkyl_reader import GkylReader
from postgkyl.data.gkyl_adios_reader import GkylAdiosReader
from postgkyl.data.gkyl_h5_reader import GkylH5Reader
from postgkyl.data.flash_h5_reader import FlashH5Reader
import postgkyl.utils.gkeyll_enums as gkenums


class GData(object):
  """Provides interface to (not only) Gkeyll output data.

  GData serves as a baseline interface to Gkeyll data. It is used for
  loading Gkeyll data and serves is input to many Postgkyl
  functions. Represents a dataset in the Postgkyl command line mode.

  Examples:
    import postgkyl as pg
    data = pg.GData('file.gkyl', comp=1)

  """

  def __init__(self, file_name: str = "",
      comp: int | str | None = None,
      z0: int | str | None = None, z1: int | str | None = None,
      z2: int | str | None = None, z3: int | str | None = None,
      z4: int | str | None = None, z5: int | str | None = None,
      var_name: str = "CartGridField",
      tag: str = "default", label: str = "",
      ctx: dict | None = None,
      comp_grid: bool = False, mapc2p_name: str = "", mapc2p_vel_name: str = "",
      reader_name: str = "", load: bool = True, click_mode: bool = False):
    """Initializes the Data class with a Gkeyll output file.

    Args:
      fileName: str
        The name of Gkeyll output file. Currently supported are 'h5',
        ADIOS 'bp', and binary 'gkyl' files. Can be ommited for empty
        class.
      comp: int or 'int:int'
        Load only the specified component index or a slice of
        idices. Supported only for the ADIOS 'bp' files.
      z0 - z5: int or 'int:int'
        Load only the specified  index or a slice of
        idices in a direction. Supported only for the ADIOS 'bp' files.
      var_name: str
        Specify custom ADIOS variable name (default is 'CartGridField').
      tag: str
        Specify dataset tag for use in the command line mode.
      label: str
        Specify dataset label for use in the command line mode.
      ctx: dict
        Copy content of the specified ctx dictionary.
      comp_grid: bool
        A flag to ignore grid mapping.
      mapc2p_name: str
        The name of the file containg the c2p mapping.
      mapc2p_vel_name: str
        The name of the file containg the c2p mapping just for velocity.
      reader_name: str
        Reader can be specified to bypass the automatic selection.
      load: bool = True
        Automatically the data to memory; when set to False, data can be loaded later
        using the load() method.
      click_mode: bool = False
        Enables command-line behavior like prompting when a
        var_name is either missing or doesn't match any available.
    """
    self._grid = None
    self._values = None  # (N+1)D narray of values

    # Context dictionary to store metadata, filled by the reader.
    self.ctx = {}
    
    # Allow to copy input context variable
    if ctx:
      for key in ctx:
        self.ctx[key] = ctx[key]

    self._tag = tag
    self._comp_grid = comp_grid  # flag to disregard the mapped grid
    self._label = ""
    self._custom_label = label
    self._var_name = var_name
    self._file_name = str(file_name)
    self._mapc2p_name = mapc2p_name
    self._mapc2p_vel_name = mapc2p_vel_name
    self.color = None

    self._neighbors = []

    self._status = True

    zs = (z0, z1, z2, z3, z4, z5)

    readers = {
      "gkyl": GkylReader,
      "adios": GkylAdiosReader,
      "h5": GkylH5Reader,
      "flash": FlashH5Reader,
    }
    if self._file_name:
      reader_set = False
      if reader_name in readers:
        # Keep only the user-specified reader
        reader = readers[reader_name]
        readers.clear()
        readers[reader_name] = reader
      # end
      for key, rd in readers.items():
        self._reader = rd(file_name=self._file_name, ctx=self.ctx, var_name=var_name,
          c2p=mapc2p_name, c2p_vel=mapc2p_vel_name, axes=zs, comp=comp,
          click_mode=click_mode)
        if self._reader.is_compatible():
          reader_set = True
          break
        # end
      # end
      if not reader_set:
        raise NameError(f"'file_name' was specified ({self._file_name}) but cannot be read with {list(readers)}")
      # end

      self._reader.preload()
      if load:
        self._grid, self._values = self._reader.load()
      # end
    # end

  # ---- Tag ----
  def get_tag(self) -> str:
    return self._tag

  def set_tag(self, tag: str = "") -> None:
    if tag:
      self._tag = tag
    # end

  tag = property(get_tag, set_tag)

  # ---- Label ----
  def get_label(self) -> str:
    if self._custom_label:
      return self._custom_label
    else:
      return self._label
    # end

  def set_label(self, label: str) -> None:
    self._label = label

  label = property(get_label, set_label)

  def get_custom_label(self):
    return self._custom_label

  # ---- Status ----
  def activate(self) -> None:
    self._status = True

  def deactivate(self) -> None:
    self._status = False

  def get_status(self) -> bool:
    return self._status

  status = property(get_status)

  # ---- Input file ----
  def get_input_file(self) -> str:
    if not has_adios:
      raise ModuleNotFoundError("ADIOS2 is not installed")
    # end

    fh = adios2.open(self._file_name, "rra")
    input_file = fh.read_attribute_string("inputfile")[0]
    fh.close()
    return input_file

  # ---- Number of Cells ----
  def get_num_cells(self) -> np.ndarray:
    if self.ctx.get("cells") is not None:
      return self.ctx["cells"]
    elif self._values is not None:
      num_dims = len(self._values.shape) - 1
      cells = np.zeros(num_dims, np.int32)
      for d in range(num_dims):
        cells[d] = int(self._values.shape[d])
      # end
      return cells
    else:
      return 0
    # end

  num_cells = property(get_num_cells)

  # ---- Number of Components ----
  def get_num_comps(self) -> int:
    if self.ctx.get("num_comps"):
      return self.ctx["num_comps"]
    elif self._values is not None:
      return int(self._values.shape[-1])
    else:
      return 0
    # end

  num_comps = property(get_num_comps)

  # ---- Number of Dimensions -----
  def get_num_dims(self, squeeze: bool = False) -> int:
    if self.ctx.get("cells") is not None:
      num_dims = len(self.ctx["cells"])
    elif self._values is not None:
      num_dims = int(len(self._values.shape) - 1)
    else:
      return 0
    # end
    if squeeze:
      cells = self.get_num_cells()
      for d in range(num_dims):
        if cells[d] == 1:
          num_dims = num_dims - 1
        # end
      # end
    # end
    return num_dims

  num_dims = property(get_num_dims)

  # ---- Grid Bounds ----
  def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    if "lower" in self.ctx.keys() and "upper" in self.ctx.keys():
      return self.ctx["lower"], self.ctx["upper"]
    elif self._grid is not None:
      num_dims = len(self._values.shape) - 1
      lo, up = np.zeros(num_dims), np.zeros(num_dims)
      for d in range(num_dims):
        lo[d] = self._grid[d].min()
        up[d] = self._grid[d].max()
      # end
      return lo, up
    else:
      return None, None
    # end

  bounds = property(get_bounds)

  # ---- Grid and Values ----
  def get_grid(self) -> list:
    return self._grid

  def set_grid(self, grid: list) -> None:
    self._grid = grid
    num_dims = self.get_num_dims()
    lo, up = np.zeros(num_dims), np.zeros(num_dims)
    for d in range(num_dims):
      lo[d] = self._grid[d].min()
      up[d] = self._grid[d].max()
    self.ctx["lower"] = lo
    self.ctx["upper"] = up

  grid = property(get_grid, set_grid)

  def get_grid_type(self) -> str:
    return self.ctx["grid_type"]

  def get_values(self) -> np.ndarray:
    return self._values

  def set_values(self, values) -> None:
    self._values = values
    if "cells" not in self.ctx or not np.array_equal(values.shape[:-1], self.ctx["cells"]):
      self.ctx["cells"] = values.shape[:-1]
    if "num_comps" not in self.ctx or values.shape[-1] != self.ctx["num_comps"]:
      self.ctx["num_comps"] = values.shape[-1]

  values = property(get_values, set_values)

  def push(self, grid, values):
    self.set_values(values)
    self.set_grid(grid)
    return self

  # ---- Neighboring Blocks ----
  def set_neighbors(self, dataspace):
    data_list = list(dataspace)
    num_dims = self.get_num_dims()
    for dim in range(num_dims):
      self._neighbors.append([None, None])
      for data in data_list:
        if num_dims == 1:
          if np.isclose(self.get_grid()[dim][0], data.get_grid()[dim][-1]):
            self._neighbors[dim][0] = data
          elif np.isclose(self.get_grid()[dim][-1], data.get_grid()[dim][0]):
            self._neighbors[dim][1] = data
        elif num_dims == 2:
          if np.isclose(self.get_grid()[dim][0], data.get_grid()[dim][-1]) and np.isclose(self.get_grid()[not dim][0], data.get_grid()[not dim][0]):
            self._neighbors[dim][0] = data
          elif np.isclose(self.get_grid()[dim][-1], data.get_grid()[dim][0]) and np.isclose(self.get_grid()[not dim][0], data.get_grid()[not dim][0]):
            self._neighbors[dim][1] = data
        elif num_dims == 3:
          rem_dims = list(range(num_dims)).remove(dim)
          if np.isclose(self.get_grid()[dim][0], data.get_grid()[dim][-1]) and np.isclose(self.get_grid()[rem_dims[0]][0], data.get_grid()[rem_dims[0]][0]) and np.isclose(self.get_grid()[rem_dims[1]][0], data.get_grid()[rem_dims[1]][0]):
            self._neighbors[dim][0] = data
          elif np.isclose(self.get_grid()[dim][0], data.get_grid()[dim][-1]) and np.isclose(self.get_grid()[rem_dims[0]][0], data.get_grid()[rem_dims[0]][0]) and np.isclose(self.get_grid()[rem_dims[1]][0], data.get_grid()[rem_dims[1]][0]):
            self._neighbors[dim][1] = data
        # end
      # end
    # end

  def __dict_has_key_from_group__(self, dict_in, group_members_in):
    # Check if a dictionary with key-value pairs, where the key is the name of a group and
    # the value a list of group members (as strings), has a member from a given
    # group.
    #  dict_in: input dictionary.
    #  group_members_in: group members to check for in dict_in.
    dict_keys = dict_in.keys()
    for key in group_members_in:
      if key in dict_keys:
        return True
      #end
    #end
    return False
      

  # ---- Info -----
  def info(self) -> str:
    """Prints GData object information.

    Prints time (only when available), number of components, dimension
    spans, extremes for a GData object.

    Args:
      none

    Returns:
      output: str
        A list of strings with the informations
    """
    values = self.values
    num_comps = self.num_comps
    num_dims = self.num_dims
    num_cells = self.num_cells
    lower, upper = self.bounds

    # Groups of metadata.
    info_groups = {
      "time_info" : ["time","frame"],
      "grid_info" : ["lower","upper","cells","grid_type"],
      "basis_info" : ["poly_order","basis_type","is_modal","num_comps"],
      "build_info" : ["changeset","builddate"], 
      "geometry_info": ["geometry_type", "geqdsk_sign_convention"],
    }

    output = ""
    
    printed_keys = []

    if "time" in self.ctx.keys():
      printed_keys.append("time")
      output += f"├─ Time: {self.ctx['time']:e}\n"
    # end

    if "frame" in self.ctx.keys():
      printed_keys.append("frame")
      output += f"├─ Frame: {self.ctx['frame']:d}\n"
    # end

    output += f"├─ Number of components: {num_comps:d}\n"
    output += f"├─ Number of dimensions: {num_dims:d}\n"
    if self.__dict_has_key_from_group__(self.ctx, info_groups["grid_info"]):
      output += f"├─ Grid: ({self.get_grid_type():s})\n"
      if "lower" in self.ctx.keys() and "upper" in self.ctx.keys() and "cells" in self.ctx.keys():
        for d in range(num_dims - 1):
          output += f"│  ├─ Dim {d:d}: Num. cells: {num_cells[d]:d}; "
          output += f"Lower: {lower[d]:e}; Upper: {upper[d]:e}\n"
        # end
      # end

      output += f"│  └─ Dim {num_dims - 1:d}: Num. cells: {num_cells[-1]:d}; "
      output += f"Lower: {lower[-1]:e}; Upper: {upper[-1]:e}"
    # end

    if values is not None:
      maximum = np.nanmax(values)
      max_idx = np.unravel_index(np.nanargmax(values), values.shape)
      minimum = np.nanmin(values)
      min_idx = np.unravel_index(np.nanargmin(values), values.shape)
      output += f"\n├─ Maximum: {maximum:e} at {str(max_idx[:num_dims]):s}"
      if num_comps > 1:
        output += f" component {max_idx[-1]:d}\n"
      else:
        output += "\n"
      # end
      output += f"├─ Minimum: {minimum:e} at {str(min_idx[:num_dims]):s}"
      if num_comps > 1:
        output += f" component {min_idx[-1]:d}"
      # end
    # end

    if self.__dict_has_key_from_group__(self.ctx, info_groups["basis_info"]):
      output += "\n├─ DG info:"
      if "poly_order" in self.ctx.keys():
        printed_keys.append("poly_order")
        output += f"\n│  ├─ Polynomial Order: {self.ctx['poly_order']:d}"
      # end
      if "basis_type" in self.ctx.keys():
        printed_keys.append("basis_type")
        if self.ctx["is_modal"]:
          output += f"\n│  └─ Basis Type: {self.ctx['basis_type']:s} (modal)"
        else:
          output += f"\n│  └─ Basis Type: {self.ctx['basis_type']:s}"
        # end
      # end
    # end

    if self.__dict_has_key_from_group__(self.ctx, info_groups["build_info"]):
      output += "\n├─ Created with Gkeyll:"
      if "changeset" in self.ctx.keys():
        printed_keys.append("changeset")
        output += f"\n│  ├─ Changeset: {self.ctx['changeset']:s}"
      # end
      if "builddate" in self.ctx.keys():
        printed_keys.append("builddate")
        output += f"\n│  └─ Build Date: {self.ctx['builddate']:s}"
      # end
    # end

    if self.__dict_has_key_from_group__(self.ctx, info_groups["geometry_info"]):
      output += "\n├─ Geometry info:"
      if "geometry_type" in self.ctx.keys():
        printed_keys.append("geometry_type")
        output += f"\n│  ├─ Type: {gkenums.gkyl_geometry_id[self.ctx['geometry_type']]:s}"
      # end
      if "geqdsk_sign_convention" in self.ctx.keys():
        printed_keys.append("geqdsk_sign_convention")
        output += f"\n│  ├─ GEQDSK sign convention: {self.ctx['geqdsk_sign_convention']:d}"
      # end
    # end
      
    # Print any other keys in the context that were not printed above
    for key, val in self.ctx.items():
      if key not in sum(info_groups.values(), []):
        output += f"\n├─ {key:s}: {val}"
      # end
    # end

    return output

  # ---- Write ----
  def write(self, out_name: str = "",
      extension: Literal["gkyl", "bp", "txt", "npy"] = "gkyl",
      mode: str = "", var_name: str = "", append: bool = False,
      cleaning: bool = True) -> None:
    """Writes data in a file.

    The available formats are Gkeyll .gkyl (default), ADIOS .bp file, ASCII .txt file,
    or NumPy .npy file.

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

  # ---- Context (metadata) ----
  def get_ctx(self) -> dict:
    return self.ctx

