import numpy as np
import tables


class Read_gkyl_h5(object):
  """Provides a framework to read gkyl Adios output"""

  def __init__(self, file_name: str, ctx: dict = None, **kwargs) -> None:
    self._file_name = file_name

    self.is_frame = False
    self.is_diagnostic = False

    self.ctx = ctx

  # end

  def _is_compatible(self) -> bool:
    try:
      fh = tables.open_file(self._file_name, "r")

      if "/DataStruct/data" in fh:
        self.is_diagnostic = True
      # end
      if "/StructGridField" in fh:
        self.is_frame = True
      # end

      fh.close()
    except:
      return False
    # end
    return self.is_frame or self.is_diagnostic

  # end

  def _read_frame(self) -> tuple:
    fh = tables.open_file(self._file_name, "r")

    # Postgkyl conventions require the attributes to be
    # narrays even for 1D data
    lower = np.atleast_1d(fh.root.StructGrid._v_attrs.vsLowerBounds)
    upper = np.atleast_1d(fh.root.StructGrid._v_attrs.vsUpperBounds)
    cells = np.atleast_1d(fh.root.StructGrid._v_attrs.vsNumCells)
    if "/timeData" in fh:
      self.ctx["time"] = fh.root.timeData._v_attrs.vsTime
    # end

    data = fh.root.StructGridField.read()

    fh.close()
    return cells, lower, upper, data

  def _read_diagnostic(self):
    fh = tables.open_file(self._file_name, "r")

    grid = fh.root.DataStruct.timeMesh.read()
    data = fh.root.DataStruct.data.read()

    fh.close()
    # end

    return [np.squeeze(grid)], [grid[0]], [grid[-1]], data

  # end

  # ---- Exposed function ----------------------------------------------
  def get_data(self) -> tuple:
    grid = None

    if self.is_frame:
      cells, lower, upper, data = self._read_frame()
    # end
    if self.is_diagnostic:
      grid, lower, upper, data = self._read_diagnostic()
      cells = grid[0].shape
    # end

    num_dims = len(cells)
    grid = [np.linspace(lower[d], upper[d], cells[d] + 1) for d in range(num_dims)]
    if self.ctx:
      self.ctx["grid_type"] = "uniform"
    # end

    return grid, data

  # end
