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


class Read_gkyl(object):
  """Provides a framework to read gkylzero binary output
  """

  def __init__(self, file_name : str,
               ctx : dict = None,
               c2p : str = None,
               **kwargs) -> None:
    self.file_name = file_name
    self.c2p = c2p

    self.dtf = np.dtype('f8')
    self.dti = np.dtype('i8')

    self.offset = 0
    self.doffset = 8

    self.file_type = 1
    self.version = 0

    self.ctx = ctx
  #end

  def _is_compatible(self) -> bool:
    try:
      magic = np.fromfile(self.file_name, dtype=np.dtype('b'),
                          count=5, offset=0)
      if np.array_equal(magic, [103, 107, 121, 108, 48]):
        self.version = np.fromfile(self.file_name, dtype=self.dti,
                                   count=1, offset=5)[0]
        return True
      #end
    except:
      return False
    #end
    return False
  #end

  # Starting with version 1, .gkyl files contatin a header; version 0
  # files only include the real-type info
  def get_meta(self) -> None:
    self.offset = 0

    if self._is_compatible():
      self.offset += 5 # Header contatins the gkyl magic sequence

      self.version = np.fromfile(self.file_name, dtype=self.dti,
                                 count=1, offset=self.offset)[0]
      self.offset += 8

      self.file_type = np.fromfile(self.file_name, dtype=self.dti,
                                   count=1, offset=self.offset)[0]
      self.offset += 8

      meta_size = np.fromfile(self.file_name, dtype=self.dti,
                              count=1, offset=self.offset)[0]
      self.offset += 8

      # read meta
      self.offset += meta_size # Skip this for now
    #end

    # read real-type
    real_type = np.fromfile(self.file_name, dtype=self.dti, count=1,
                            offset=self.offset)[0]
    if real_type == 1:
      self.dtf = np.dtype('f4')
      self.doffset = 4
    #end
    self.offset += 8
  #end

  # ---- Read field data (version 1) -----------------------------------
  def _read_domain_t1a3_v1(self) -> None:
    # read grid dimensions
    self.num_dims = np.fromfile(self.file_name, dtype=self.dti,
                                count=1, offset=self.offset)[0]
    self.offset += 8

    # read grid shape
    self.cells = np.fromfile(self.file_name, dtype=self.dti,
                             count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * 8

    # read lower/upper
    self.lower = np.fromfile(self.file_name, dtype=self.dtf,
                             count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * self.doffset
    self.upper = np.fromfile(self.file_name, dtype=self.dtf,
                        count=self.num_dims, offset=self.offset)
    self.offset += self.num_dims * self.doffset

    # read array elem_ez (the div by doffset is as elem_sz includes
    # sizeof(real_type) = doffset)
    elem_sz_raw = int(
      np.fromfile(self.file_name, dtype=self.dti,
                  count=1, offset=self.offset)[0])
    elem_sz = elem_sz_raw / self.doffset
    self.num_comps = int(elem_sz)
    self.offset += 8

    # read array size
    self.asize = np.fromfile(self.file_name, dtype=self.dti,
                             count=1, offset=self.offset)[0]
    self.offset += 8

  def _read_data_t1_v1(self) -> np.ndarray:
    data_raw = np.fromfile(self.file_name, dtype=self.dtf,
                           offset=self.offset)
    gshape = np.ones(self.num_dims+1, dtype=self.dti)
    for d in range(self.num_dims):
      gshape[d] = self.cells[d]
    #end
    gshape[-1] = self.num_comps
    return data_raw.reshape(gshape)
  #end

  def _read_data_t3_v1(self) -> np.ndarray:
    # get the number of stored ranges
    num_range = np.fromfile(self.file_name, dtype=self.dti,
                            count=1, offset=self.offset)[0]
    self.offset += 8

    gshape = np.ones(self.num_dims+1, dtype=self.dti)
    gshape[-1] = self.num_comps

    data = np.zeros(gshape, dtype=self.dtf)
    for i in range(num_range):
      loidx = np.fromfile(self.file_name, dtype=self.dti,
                          count=self.num_dims, offset=self.offset)
      self.offset += self.num_dims * 8
      upidx = np.fromfile(self.file_name, dtype=self.dti,
                          count=self.num_dims, offset=self.offset)
      self.offset += self.num_dims * 8
      for d in range(self.num_dims):
        gshape[d] = upidx[d] - loidx[d] + 1
      #end
      slices = [slice(loidx[d]-1,upidx[d]) for d in range(self.num_dims)]

      asize = np.fromfile(self.file_name, dtype=self.dti,
                          count=1, offset=self.offset)[0]
      self.offset += 8
      data_raw = np.fromfile(self.file_name, dtype=self.dtf,
                             count=asize*self.num_comps,
                             offset=self.offset)
      self.offset += asize * self.num_comps * self.doffset
      data[tuple(slices)] = data_raw.reshape(gshape)
    #end
    return data
  #end

  # ---- Read dynvector data (version 1) -------------------------------
  def _read_t2_v1(self) -> tuple:
    cells = [0]
    time = np.array([])
    data = np.array([[]])
    while True: # Python does not have DO .. WHILE loop
      elem_sz_raw = int(
        np.fromfile(self.file_name, dtype=self.dti,
                    count=1, offset=self.offset)[0])
      num_comps = int(elem_sz_raw / self.doffset)
      self.offset += 8

      loop_cells = int(np.fromfile(self.file_name, dtype=self.dti,
                                   count=1, offset=self.offset)[0])
      self.offset += 8

      loop_time = np.fromfile(self.file_name, dtype=self.dtf,
                              count=cells, offset=self.offset)
      self.offset += loop_cells * 8

      data_raw = np.fromfile(self.file_name, dtype=self.dtf,
                             count=num_comps * cells,
                             offset=self.offset)
      self.offset += loop_cells * elem_sz_raw
      gshape = np.array((loop_cells, num_comps), dtype=self.dti)

      time = np.append(time, loop_time)
      if cells[0] == 0:
        data = data_raw.reshape(gshape)
      else:
          data = np.append(data, data_raw.reshape(gshape), axis=0)
      #end
      cells[0] += loop_cells[0]
      if self.offset == os.path.getsize(self.file_name):
        break
      #end
      self._read_header()
      if self.file_type != 2:
        raise TypeError('Inconsitent data in g0 dynVector file.')
      #end
      self.cells = cells
      self.lower = np.atleast_1d(time.min())
      self.upper = np.atleast_1d(time.max())
      return time, data
    #end
  #end

  # ---- Exposed function ----------------------------------------------
  def get_data(self) -> tuple:
    time = None
    if self.file_type == 1 or self.version == 0:
      self._read_domain_t1a3_v1()
      data = self._read_data_t1_v1()
    elif self.file_type == 2:
      time, data = self._read_t2_v1()
    elif self.file_type == 3:
      self._read_domain_t1a3_v1()
      data = self._read_data_t3_v1()
    else:
      raise TypeError('This g0 format is not presently supported')
    #end


    # Load or construct grid
    if time is not None:
      grid = [time]
      if self.ctx:
        self.ctx['grid_type'] = 'nodal'
      #end
    elif self.c2p:
      grid_reader = Read_gkyl(self.c2p)
      _, tmp = grid_reader.get_data()
      num_comps = tmp.shape[-1]
      num_coeff = num_comps/num_dims
      grid = [tmp[..., int(d*num_coeff):int((d+1)*num_coeff)]
              for d in range(num_dims)]
      if self.ctx:
        self.ctx['grid_type'] = 'c2p'
      #end
    else: # Create sparse unifrom grid
      num_dims = len(self.cells)
      # Adjust for ghost cells
      dz = (self.upper - self.lower) / self.cells
      for d in range(num_dims):
        if self.cells[d] != data.shape[d]:
          ngl = int(np.floor((self.cells[d] - data.shape[d])*0.5))
          ngu = int(np.ceil((self.cells[d] - data.shape[d])*0.5))
          self.cells[d] = data.shape[d]
          self.lower[d] = self.lower[d] - ngl*dz[d]
          self.upper[d] = self.upper[d] + ngu*dz[d]
        #end
      #end
      grid = [np.linspace(self.lower[d],
                          self.upper[d],
                          self.cells[d]+1)
              for d in range(num_dims)]
      if self.ctx:
        self.ctx['grid_type'] = 'uniform'
      #end
    #end

    return grid, data
  #end
#end