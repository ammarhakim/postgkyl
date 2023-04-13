import numpy as np
import os.path

# Format description for raw Gkeyll output file from
# gkyl_array_rio_format_desc.h

# The format of the gkyl binary output is as follows.

# ## Version 0: Jan 2021. Created by A.H. Note Version 0 has no header
#    information

# Data      Type and meaning
# --------------------------
# ndim      uint64_t Dimension of field
# cells     uint64_t[ndim] number of cells in each direction
# lower     float64[ndim] Lower bounds of grid
# upper     float64[ndim] Upper bounds of grid
# esznc     uint64_t Element-size * number of components in field
# size      uint64_t Total number of cells in field
# DATA      size*esznc bytes of data

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

def _is_gkyl(file_name : str, offset : int) -> bool:
  magic = np.fromfile(file_name, dtype=np.dtype('b'), count=5, offset=offset)
  if np.array_equal(magic, [103, 107, 121, 108,  48]):
    return True
  else:
    return False
  #end
#end

def _load_header(file_name : str, offset : int):
  file_type = 1
  version = 0
  dtf = np.dtype('f8')
  doffset = 8
  dti = np.dtype('i8')
  
  if _is_gkyl(file_name, offset): # Check if version >= 1
    offset += 5
    
    version = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8
    
    file_type = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8
  
    meta_size = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8

    # read meta
    offset += meta_size # Skip this for now
  #end

  # read real-type
  real_type = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  if real_type == 1:
    dtf = np.dtype('f4')
    doffset = 4
  #end
  offset += 8
  return version, file_type, offset, dtf, doffset
#end

def _load_t1_v1(file_name, offset, dti, dtf, doffset):
  # read grid dimensions
  num_dims = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  offset += 8

  # read grid shape
  cells = np.fromfile(file_name, dtype=dti, count=num_dims, offset=offset)
  offset += num_dims*8

  # read lower/upper
  lower = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
  offset += num_dims*doffset
      
  upper = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
  offset += num_dims*doffset

  # read array elemEz (the div by doffset is as elemSz includes
  # sizeof(real_type) = doffset)
  elemSzRaw = int(
    np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0])
  elemSz = elemSzRaw/doffset
  offset += 8
    
  # read array size
  asize = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  offset += 8

  adata = np.fromfile(file_name, dtype=dtf, offset=offset)
  gshape = np.ones(num_dims+1, dtype=dti)
  for d in range(num_dims):
    gshape[d] = cells[d]
  #end
  num_comp = int(elemSz)
  gshape[-1] = num_comp
  return offset, num_dims, cells, lower, upper, adata.reshape(gshape)
#end

def _load_t2_v1(file_name, offset, dti, dtf, doffset):
  elemSzRaw = int(
    np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0])
  num_comp = int(elemSzRaw/doffset)
  offset += 8

  cells = int(np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0])
  offset += 8

  time = np.fromfile(file_name, dtype=dtf, count=cells, offset=offset)
  offset += cells * 8

  adata = np.fromfile(file_name, dtype=dtf, count=num_comp*cells, offset=offset)
  offset += cells * elemSzRaw
  gshape = np.ones(2, dtype=dti)
  gshape[0] = cells
  gshape[1] = num_comp
  return offset, [cells], time, adata.reshape(gshape)
#end

def _load_t3_v1(file_name, offset, dti, dtf, doffset):
  # read grid dimensions
  num_dims = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  offset += 8

  # read grid shape
  cells = np.fromfile(file_name, dtype=dti, count=num_dims, offset=offset)
  offset += num_dims * 8

  # read lower/upper
  lower = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
  offset += num_dims * doffset
      
  upper = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
  offset += num_dims * doffset

  # read array elemEz (the div by doffset is as elemSz includes
  # sizeof(real_type) = doffset)
  elemSzRaw = int(
    np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0])
  elemSz = elemSzRaw / doffset
  offset += 8
    
  # read array size
  asize = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  offset += 8

  # get the number of stored ranges
  num_range = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  offset += 8

  num_comp = int(elemSz)
  gshape = np.ones(num_dims+1, dtype=dti)
  for d in range(num_dims):
    gshape[d] = cells[d]
  #end
  gshape[-1] = num_comp
  data = np.zeros(gshape, dtype=dtf)
  for i in range(num_range):
    loidx = np.fromfile(file_name, dtype=dti, count=num_dims, offset=offset)
    offset += num_dims * 8
    upidx = np.fromfile(file_name, dtype=dti, count=num_dims, offset=offset)
    offset += num_dims * 8
    for d in range(num_dims):
      gshape[d] = upidx[d] - loidx[d] + 1
    #end
    slices = [slice(loidx[d]-1,upidx[d]) for d in range(num_dims)]
    
    asize = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8
    adata = np.fromfile(file_name, dtype=dtf, count=asize*num_comp, offset=offset)
    offset += asize * num_comp * doffset
    data[tuple(slices)] = adata.reshape(gshape)
  #end
  return offset, num_dims, cells, lower, upper, data
#end

def load_gkyl(file_name : str) -> tuple:
  dti = np.dtype('i8')
  offset = 0

  version, file_type, offset, dtf, doffset = _load_header(
    file_name, offset)
  if file_type == 1 or version == 0:
    offset, num_dims, cells, lower, upper, data = _load_t1_v1(
      file_name, offset, dti, dtf, doffset)
    return num_dims, cells, (lower, upper), data
  elif file_type == 3:
    offset, num_dims, cells, lower, upper, data = _load_t3_v1(
      file_name, offset, dti, dtf, doffset)
    return num_dims, cells, (lower, upper), data
  elif file_type == 2:
    cells = [0]
    time = np.array([])
    data = np.array([[]])
    while True:
      offset, cells1, time1, data1 = _load_t2_v1(
        file_name, offset, dti, dtf, doffset)
      time = np.append(time, time1)
      if cells[0] == 0:
        data = data1
      else:
        data = np.append(data, data1, axis=0)
      #end
      cells[0] += cells1[0]
      if offset == os.path.getsize(file_name):
        break
      #end
      version, file_type, offset, dtf, doffset = _load_header(
        file_name, offset)
      if file_type != 2:
        raise TypeError('Inconsitent data in g0 dynVector file.')
      #end
    #end
    return 1, cells, time, data
  else:
    raise TypeError('This g0 format is not presently supported')
  #end
#end
