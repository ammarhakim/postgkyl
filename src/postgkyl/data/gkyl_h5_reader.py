"""Module including legacy Gkeyll reader class"""

from typing import Tuple
import numpy as np
import tables


class GkylH5Reader(object):
  """Provides a framework to read legacy Gkeyll HDF5 output"""

  def __init__(self, file_name: str, ctx: dict | None = None, **kwargs):
    """Initialize the instance of Gkeyll reader.

    Args:
      file_name: str
      ctx: dict
        Passes context variable with metadata.
      **kwargs
        This is not directly used but allowes for unified interface to all the readers
        we use.
    """
    self._file_name = file_name

    self.is_frame = False
    self.is_diagnostic = False

    self.ctx = ctx

  def is_compatible(self) -> bool:
    """Checks if file can be read with the legacy Gkeyll HDF5 reader."""
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

  def preload(self) -> None:
    """Loads metadata."""
    pass

  def load(self) -> Tuple[list, np.ndarray]:
    """Loads data.

    Returns:
      A tuple including a grid list and a data NumPy array

    Notes:
      Needs to be called after the preload.
    """
    grid = None

    if self.is_frame:
      cells, lower, upper, data = self._read_frame()
    else:
      grid, lower, upper, data = self._read_diagnostic()
      cells = grid[0].shape
    # end

    if self.ctx:
      self.ctx["cells"] = cells
      self.ctx["lower"] = lower
      self.ctx["upper"] = upper
      self.ctx["num_comps"] = 1
      if len(data.shape) > len(cells):
        self.ctx["num_comps"] = data.shape[-1]
      # end
    # end

    num_dims = len(cells)
    grid = [np.linspace(lower[d], upper[d], cells[d] + 1) for d in range(num_dims)]
    if self.ctx:
      self.ctx["grid_type"] = "uniform"
    # end

    return grid, data
