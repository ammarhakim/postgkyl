"""Module including Gkeyll binary reader class."""

from typing import Tuple
import msgpack as mp
import numpy as np
import os.path

# Format description for raw Gkeyll output file from
# gkyl_array_rio_format_desc.h

# The format of the gkyl binary output is as follows.

# ----------------------------------------------------------------------
# ## Version 0: Jan 2021. Created by A.H.
#    Note Version 0 has no header information

# Data      Type and meaning
# --------------------------
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# DATA      size*esznc bytes of data

# ----------------------------------------------------------------------
# ## Version 1: May 9th 2022. Created by A.H

# Data      Type and meaning
# --------------------------
# gkyl0     5 bytes
# version   uint64_t
# file_type uint64_t (See header gkyl_elem_type.h for file types)
# meta_size uint64_t Number of bytes of meta-data
# DATA      meta_size bytes of data. This is in msgpack format

# * For file_type = 1 (field) the above header is followed by

# real_type uint64_t. Indicates real type of data
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# DATA      size*esznc bytes of data

# * For file_type = 2 (dynvec) the above header is followed by

# real_type uint64_t. Indicates real type of data
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# TIME_DATA float64[size] bytes of data
# DATA      size*esznc bytes of data

# * For file_type = 3 (multi-range field) the above header is followed by

# real_type uint64_t. Indicates real type of data
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# nrange    uint64_t Number of ranges stored in this file

# For each of the nrange ranges in the field the following data is
# present

# loidx     uint64_t[ndim] Index of lower-left corner of the range
# upidx     uint64_t[ndim] Index of upper-right corner of the range
# size      uint64_t Total number of cells in range
# DATA      size*esznc bytes of data

# Note: the global range in Gkeyll, of which each range is a part,
# is 1-indexed.


class GkylReader(object):
  """Provides a framework to read Gkeyll binary output."""

  def __init__(self, file_name: str, ctx: dict | None = None,
      c2p: str = "", c2p_vel: str = "", **kwargs):
    """Initialize the instance of Gkeyll reader.

    Args:
      file_name: str
      ctx: dict
        Passes context variable with metadata.
      var_name: str = "CartGridField"
      c2p: str
        Allows to specify a name of the file containing c2p mapping.
      c2p_vel: str
        Allows to specify a name of the file containing c2p mapping for only the
        velocity dimension.
      **kwargs
        This is not directly used but allowes for unified interface to all the readers
        we use.
    """
    self.file_name = file_name
    self.c2p = c2p
    self.c2p_vel = c2p_vel

    self.dtf = np.dtype("f8")
    self.dti = np.dtype("i8")

    self.offset = 0
    self.doffset = 8

    self.file_type = 1
    self.version = 0

    self.lower = None
    self.upper = None
    self.num_comps = None
    self.cells = None

    if ctx is not None:
      self.ctx = ctx
    else:
      self.ctx = {}
    # end

  def is_compatible(self) -> bool:
    """Checks if file can be read with Gkeyll reader."""
    try:
      magic = np.fromfile(self.file_name, dtype=np.dtype("b"), count=5, offset=0)
      if np.array_equal(magic, [103, 107, 121, 108, 48]):
        self.version = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=5)[0]
        return True
      else:
        return False
      # end
    except:
      return False
    #end

  # Starting with version 1, .gkyl files contatin a header; version 0
  # files only include the real-type info
  def _read_header(self) -> None:
    if self.is_compatible():
      self.offset += 5  # Header contatins the gkyl magic sequence

      self.version = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
      self.offset += 8

      self.file_type = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
      self.offset += 8

      meta_size = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
      self.offset += 8

      # read meta
      if meta_size > 0:
        fh = open(self.file_name, "rb")
        fh.seek(self.offset)
        unp = mp.unpackb(fh.read(meta_size))
        for key in unp:
          if self.ctx:
            if key == "polyOrder":
              self.ctx["poly_order"] = unp[key]
            elif key == "basisType":
              self.ctx["basis_type"] = unp[key]
              self.ctx["is_modal"] = True
            else:
              self.ctx[key] = unp[key]
            # end
          # end
        # end
        self.offset += meta_size
        fh.close()
      # end
    # end

    # read real-type
    real_type = np.fromfile(
        self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
    if real_type == 1:
      self.dtf = np.dtype("f4")
      self.doffset = 4
    # end
    self.offset += 8

  # ---- Read field data (version 1) ----
  def _read_domain_t1a3_v1(self) -> None:
    # read grid dimensions
    self.num_dims = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
    self.offset += 8

    # read grid shape
    self.cells = np.fromfile(self.file_name, dtype=self.dti, count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * 8

    # read lower/upper
    self.lower = np.fromfile(self.file_name, dtype=self.dtf, count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * self.doffset
    self.upper = np.fromfile(self.file_name, dtype=self.dtf, count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * self.doffset

    # read array elem_ez (the div by doffset is as elem_sz includes
    # sizeof(real_type) = doffset)
    elem_sz_raw = int(np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0])
    elem_sz = elem_sz_raw / self.doffset
    self.num_comps = int(elem_sz)
    self.offset += 8

    # read array size
    self.asize = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
    self.offset += 8

  def _read_data_t1_v1(self) -> np.ndarray:
    data_raw = np.fromfile(self.file_name, dtype=self.dtf, offset=self.offset)
    gshape = np.ones(self.num_dims + 1, dtype=self.dti)
    for d in range(self.num_dims):
      gshape[d] = self.cells[d]
    # end
    gshape[-1] = self.num_comps
    return data_raw.reshape(gshape)

  def _read_data_t3_v1(self) -> np.ndarray:
    # get the number of stored ranges
    num_range = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
    self.offset += 8

    gshape = np.ones(self.num_dims + 1, dtype=self.dti)
    for d in range(self.num_dims):
      gshape[d] = self.cells[d]
    # end
    gshape[-1] = self.num_comps

    data = np.zeros(gshape, dtype=self.dtf)
    for _ in range(num_range):
      loidx = np.fromfile(self.file_name, dtype=self.dti, count=self.num_dims, offset=self.offset)
      self.offset += self.num_dims * 8
      upidx = np.fromfile(self.file_name, dtype=self.dti, count=self.num_dims, offset=self.offset)
      self.offset += self.num_dims * 8
      for d in range(self.num_dims):
        gshape[d] = upidx[d] - loidx[d] + 1
      # end
      slices = [slice(loidx[d] - 1, upidx[d]) for d in range(self.num_dims)]

      asize = np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0]
      self.offset += 8
      data_raw = np.fromfile(self.file_name, dtype=self.dtf, count=asize * self.num_comps,
          offset=self.offset)
      self.offset += asize * self.num_comps * self.doffset
      data[tuple(slices)] = data_raw.reshape(gshape)
    # end
    return data

  # ---- Read dynvector data (version 1) ----
  def _read_t2_v1(self) -> Tuple[list, np.ndarray]:
    cells = 0
    time = np.array([])
    data = np.array([[]])
    while True:  # Python does not have DO .. WHILE loop
      elem_sz_raw = int(np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0])
      num_comps = int(elem_sz_raw / self.doffset)
      self.offset += 8

      loop_cells = int(np.fromfile(self.file_name, dtype=self.dti, count=1, offset=self.offset)[0])
      self.offset += 8

      loop_time = np.fromfile(self.file_name, dtype=self.dtf, count=loop_cells, offset=self.offset)
      self.offset += loop_cells * 8

      data_raw = np.fromfile(self.file_name, dtype=self.dtf, count=num_comps * loop_cells,
          offset=self.offset)
      self.offset += loop_cells * elem_sz_raw
      gshape = np.array((loop_cells, num_comps), dtype=self.dti)

      time = np.append(time, loop_time)
      if cells == 0:
        data = data_raw.reshape(gshape)
      else:
        data = np.append(data, data_raw.reshape(gshape), axis=0)
      # end
      cells += loop_cells
      if self.offset >= os.path.getsize(self.file_name):
        break
      # end
      self._read_header()
      if self.file_type != 2:
        raise TypeError("Inconsitent data in g0 dynVector file.")
      # end
    # end
    self.cells = [cells]
    self.lower = np.atleast_1d(time.min())
    self.upper = np.atleast_1d(time.max())
    return time, data

  # ---- Exposed functions -----
  def preload(self) -> None:
    """Loads metadata."""
    self._read_header()
    if self.file_type == 1 or self.file_type == 3 or self.version == 0:
      self._read_domain_t1a3_v1()
      if self.ctx:
        self.ctx["cells"] = self.cells
        self.ctx["lower"] = self.lower
        self.ctx["upper"] = self.upper
        self.ctx["num_comps"] = self.num_comps
      # end
    # end

  def load(self) -> Tuple[list, np.ndarray]:
    """Loads data.

    Returns:
      A tuple including a grid list and a data NumPy array

    Notes:
      Needs to be called after the preload.
    """
    time = None
    if self.file_type == 1 or self.version == 0:
      data = self._read_data_t1_v1()
    elif self.file_type == 2:
      time, data = self._read_t2_v1()
    elif self.file_type == 3:
      data = self._read_data_t3_v1()
    else:
      raise TypeError("This g0 format is not presently supported")
    # end

    # Load or construct grid
    num_dims = len(self.cells)
    if time is not None:
      grid = [time]
      if self.ctx:
        self.ctx["grid_type"] = "nodal"
      # end
    elif self.c2p:
      grid_reader = GkylReader(self.c2p)
      grid_reader.preload()
      _, tmp = grid_reader.load()
      num_comps = tmp.shape[-1]
      num_coeff = num_comps / num_dims
      grid = [tmp[..., int(d * num_coeff) : int((d + 1)*num_coeff)] for d in range(num_dims)]
      if self.ctx:
        self.ctx["grid_type"] = "c2p"
      # end
    elif self.c2p_vel:
      grid_reader = GkylReader(self.c2p_vel)
      grid_reader.preload()
      _, tmp = grid_reader.load()

      num_vdim = len(tmp.shape) - 1
      num_cdim = num_dims - num_vdim
      if self.ctx:
        self.ctx["num_vdim"] = num_vdim
        self.ctx["num_cdim"] = num_cdim
      # end

      # Create uniform configuration space grid
      grid = [np.linspace(self.lower[d], self.upper[d], self.cells[d] + 1) for d in range(num_cdim)]

      # Create non-uniform velocity grid
      num_comps = tmp.shape[-1]
      num_coeff = num_comps / num_vdim
      for d in range(num_vdim):
        idx = [0] * (num_vdim + 1)
        idx[d] = slice(None)
        idx[-1] = slice(int(d * num_coeff), int((d + 1) * num_coeff))
        grid.append(tmp[tuple(idx)])
      # end

      if self.ctx:
        self.ctx["grid_type"] = "c2p_vel"
      # end
    else:  # Create sparse unifrom grid
      # Adjust for ghost cells
      dz = (self.upper - self.lower) / self.cells
      for d in range(num_dims):
        if self.cells[d] != data.shape[d]:
          ngl = int(np.floor((self.cells[d] - data.shape[d]) * 0.5))
          ngu = int(np.ceil((self.cells[d] - data.shape[d]) * 0.5))
          self.cells[d] = data.shape[d]
          self.lower[d] = self.lower[d] - ngl * dz[d]
          self.upper[d] = self.upper[d] + ngu * dz[d]
        # end
      # end
      grid = [np.linspace(self.lower[d], self.upper[d], self.cells[d] + 1) for d in range(num_dims)]
      if self.ctx:
        self.ctx["grid_type"] = "uniform"
      # end
    # end

    return grid, data
