import numpy as np
import os.path

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
  
  if _is_gkyl(file_name, offset):
    offset += 5
    
    version = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8
    
    file_type = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8
  
    meta_size = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
    offset += 8

    # read meta
    offset += meta_size
  #end

  # read real-type
  realType = np.fromfile(file_name, dtype=dti, count=1, offset=offset)[0]
  if realType == 1:
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
  numComp = int(elemSz)
  gshape[-1] = numComp
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
  offset += 8*cells

  adata = np.fromfile(file_name, dtype=dtf, count=num_comp*cells, offset=offset)
  offset += cells*elemSzRaw
  gshape = np.ones(2, dtype=dti)
  gshape[0] = cells
  gshape[1] = num_comp
  return offset, [cells], time, adata.reshape(gshape)
#end

def load_gkyl(file_name : str) -> tuple:
  dti = np.dtype('i8')
  offset = 0

  version, file_type, offset, dtf, doffset = _load_header(
    file_name, offset)
  if file_type == 1 or version == 0:
    offset, num_dims, cells, lower, upper, data = _load_t1_v1(
      file_name, offset, dti, dtf, doffset)
    return num_dims, cells, lower, upper, data
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
    return 1, cells, np.array([time[0]]), np.array([time[-1]]), data
  else:
    raise TypeError('This g0 format is not presently supported')
  #end
#end
