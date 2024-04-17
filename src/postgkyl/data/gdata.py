import numpy as np
import shutil
from typing import Union

from postgkyl.data.read_gkyl import Read_gkyl
from postgkyl.data.read_gkyl_adios import Read_gkyl_adios
from postgkyl.data.read_gkyl_h5 import Read_gkyl_h5
from postgkyl.data.read_flash_h5 import Read_flash_h5

class GData(object):
  """Provides interface to Gkeyll output data.

  Data serves as a baseline interface to Gkeyll data. It is used for
  loading Gkeyll data and serves is input to many Postgkyl
  functions. Represents a dataset in the Postgkyl command line mode.

  Attributes:
    ctx: A dictionary contaioning a physics information like charge
      and/or representation information like polynomial order.
    file_name: The name of the Gkeyll file used during initialization.

  Examples:
    import postgkyl
    data = postgkyl.Data('file.bp', comp=1)

  """

  def __init__(self,
               file_name: str = '',
               comp: Union[int, str] = None,
               z0: Union[int, str] = None,
               z1: Union[int, str] = None,
               z2: Union[int, str] = None,
               z3: Union[int, str] = None,
               z4: Union[int, str] = None,
               z5: Union[int, str] = None,
               var_name: str = 'CartGridField',
               tag: str = 'default',
               label: str = '',
               ctx: dict = {},
               comp_grid: bool = False,
               mapc2p_name: str = '',
               mapc2p_vel_name: str = '',
               reader_name: str = '',
               load: bool = True,
               click_mode: bool = False) -> None:
    """Initializes the Data class with a Gkeyll output file.

    Args:
      fileName: str
        The name of Gkeyll output file. Currently supported are 'h5',
        ADIOS1 'bp', and binary 'gkyl' files. Can be ommited for empty
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
        The name of the file containg the c2p mapping information.
    """
    self._grid = None
    self._values = None # (N+1)D narray of values


    self.ctx = {}
    self.ctx['time'] = None
    self.ctx['frame'] = None
    self.ctx['lower'] = None
    self.ctx['upper'] = None
    self.ctx['cells'] = None
    self.ctx['num_cdim'] = None
    self.ctx['num_vdim'] = None
    self.ctx['num_comps'] = None
    self.ctx['changeset'] = None
    self.ctx['builddate'] = None
    self.ctx['poly_order'] = None
    self.ctx['basis_type'] = None
    self.ctx['is_modal'] = None
    self.ctx['grid_type'] = 'uniform'
    # Allow to copy input context variable
    for key in ctx:
      self.ctx[key] = ctx[key]
    #end

    self._tag = tag
    self._comp_grid = comp_grid # disregard the mapped grid
    self._label = ''
    self._custom_label = label
    self._var_name = var_name
    self._file_name = str(file_name)
    self._mapc2p_name = mapc2p_name
    self._mapc2p_vel_name = mapc2p_vel_name
    self._color = None

    self._status = True

    zs = (z0, z1, z2, z3, z4, z5)

    _readers = {
      'gkyl' : Read_gkyl,
      'adios' : Read_gkyl_adios,
      'h5' : Read_gkyl_h5,
      'flash' : Read_flash_h5
      }
    if self._file_name != '':
      reader_set = False
      if reader_name in _readers:
        # Keep only the user-specified reader
        reader = _readers[reader_name]
        _readers.clear()
        _readers[reader_name] = reader
      #end
      for key in _readers:
        self._reader = _readers[key](
          file_name = self._file_name,
          ctx = self.ctx,
          var_name = var_name,
          c2p = mapc2p_name,
          c2p_vel = mapc2p_vel_name,
          axes = zs,
          comp = comp,
          click_mode = click_mode)
        if self._reader._is_compatible():
          reader_set = True
          break
        #end
      #end
      if not reader_set:
        raise TypeError('"file_name" was specified ({:s}) but cannot be read with {:s}'.format(self._file_name, _readers))
      #end

      self._reader.preload()
      if load:
        self._grid, self._values = self._reader.load()
      #end
    #end
  #end


  #---- Stuff Control --------------------------------------------------
  def get_tag(self) -> str:
    return self._tag
  #end
  def set_tag(self, tag: str = '') -> None:
    if tag:
      self._tag = tag
    #end
  #end

  def set_label(self, label):
    self._label = label
  #end
  def get_label(self):
    if self._custom_label:
      return self._custom_label
    else:
      return self._label
    #end
  #end
  def get_custom_label(self):
    return self._custom_label
  #end

  def activate(self):
    self._status = True
  #end
  def deactivate(self):
    self._status = False
  #end
  def getStatus(self):
    return self._status
  #end

  def get_input_file(self):
    import adios2
    fh = adios2.open(self._file_name, 'rra')
    inputFile = fh.read_attribute_string('inputfile')[0]
    fh.close()
    return inputFile
  #end


  def get_num_cells(self) -> np.ndarray:
    if self.ctx['cells'] is not None:
      return self.ctx['cells']
    elif self._values is not None:
      num_dims = len(self._values.shape)-1
      cells = np.zeros(num_dims, np.int32)
      for d in range(num_dims):
        cells[d] = int(self._values.shape[d])
      #end
      return cells
    else:
      return 0
    #end
  #end

  def get_num_comps(self) -> int:
    if self.ctx['num_comps'] is not None:
      return self.ctx['num_comps']
    elif self._values is not None:
      return int(self._values.shape[-1])
    else:
      return 0
    #end
  #end

  def get_num_dims(self, squeeze=False) -> int:
    if self.ctx['cells'] is not None:
      num_dims = len(self.ctx['cells'])
    elif self._values is not None:
      num_dims = int(len(self._values.shape)-1)
    else:
      return 0
    #end
    if squeeze:
      cells = self.get_num_cells()
      for d in range(num_dims):
        if cells[d] == 1:
          num_dims = num_dims - 1
        #end
      #end
    #end
    return num_dims
  #end

  def get_bounds(self) -> np.ndarray:
    if self.ctx['lower'] is not None:
      return self.ctx['lower'], self.ctx['upper']
    elif self._grid is not None:
      num_dims = len(self._values.shape)-1
      lo, up = np.zeros(num_dims), np.zeros(num_dims)
      for d in range(num_dims):
        lo[d] = self._grid[d].min()
        up[d] = self._grid[d].max()
      #end
      return lo, up
    else:
      return None, None
    #end
  #end

  def get_grid(self) -> list:
    return self._grid
  #end

  def get_gridType(self) -> str:
    return self.ctx['grid_type']
  #end

  def get_values(self) -> np.ndarray:
    return self._values
  #end


  def set_grid(self, grid) -> None:
    self._grid = grid
    num_dims = self.get_num_dims()
    lo, up = np.zeros(num_dims), np.zeros(num_dims)
    for d in range(num_dims):
      lo[d] = self._grid[d].min()
      up[d] = self._grid[d].max()
    #end
    self.ctx['lower'] = lo
    self.ctx['upper'] = up
  #end

  def set_values(self, values) -> None:
    self._values = values
    if not np.array_equal(values.shape[:-1], self.ctx['cells']):
      self.ctx['cells'] = values.shape[:-1]
    #end
    if values.shape[-1] != self.ctx['num_comps']:
      self.ctx['num_comps'] = values.shape[-1]
    #end
  #end

  def push(self, grid, values) -> None:
    self.set_values(values)
    self.set_grid(grid)
    return self
  #end


  #---- Info -----------------------------------------------------------
  def info(self):
    """Prints Data object information.

    Prints time (only when available), number of components, dimension
    spans, extremes for a Data object.

    Args:
      none

    Returns:
      output (str): A list of strings with the informations
        """
    values = self.get_values()
    numComps = self.get_num_comps()
    num_dims = self.get_num_dims()
    numCells = self.get_num_cells()
    lower, upper = self.get_bounds()

    output = ''

    if self.ctx['time'] is not None:
      output += '├─ Time: {:e}\n'.format(self.ctx['time'])
    #end
    if self.ctx['frame'] is not None:
      output += '├─ Frame: {:d}\n'.format(self.ctx['frame'])
    #end
    output += '├─ Number of components: {:d}\n'.format(numComps)
    output += '├─ Number of dimensions: {:d}\n'.format(num_dims)
    if lower is not None:
      output += '├─ Grid: ({:s})\n'.format(self.get_gridType())
      for d in range(num_dims-1):
        output += '│  ├─ Dim {:d}: Num. cells: {:d}; '.format(d, numCells[d])
        output += 'Lower: {:e}; Upper: {:e}\n'.format(lower[d],
                                                      upper[d])
      #end
      output += '│  └─ Dim {:d}: Num. cells: {:d}; '.format(num_dims-1, numCells[num_dims-1])
      output += 'Lower: {:e}; Upper: {:e}'.format(lower[num_dims-1],
                                                  upper[num_dims-1])
    #end
    if values is not None:
      maximum = np.nanmax(values)
      maxIdx = np.unravel_index(np.nanargmax(values), values.shape)
      minimum = np.nanmin(values)
      minIdx = np.unravel_index(np.nanargmin(values), values.shape)
      output += '\n├─ Maximum: {:e} at {:s}'.format(maximum,
                                                  str(maxIdx[:num_dims]))
      if numComps > 1:
        output += ' component {:d}\n'.format(maxIdx[-1])
      else:
        output += '\n'
      #end
      output += '├─ Minimum: {:e} at {:s}'.format(minimum,
                                                 str(minIdx[:num_dims]))
      if numComps > 1:
        output += ' component {:d}'.format(minIdx[-1])
      #end
    #end
    if self.ctx['poly_order'] and self.ctx['basis_type']:
      output += '\n├─ DG info:\n'
      output += '│  ├─ Polynomial Order: {:d}\n'.format(self.ctx['poly_order'])
      if self.ctx['is_modal']:
        output += '│  └─ Basis Type: {:s} (modal)'.format(self.ctx['basis_type'])
      else:
        output += '│  └─ Basis Type: {:s}'.format(self.ctx['basis_type'])
      #end
    #end
    if self.ctx['changeset'] and self.ctx['builddate']:
      output += '\n├─ Created with Gkeyll:\n'
      output += '│  ├─ Changeset: {:s}\n'.format(self.ctx['changeset'])
      output += '│  └─ Build Date: {:s}'.format(self.ctx['builddate'])
    #end
    for key in self.ctx:
      if key not in ['time', 'frame', 'changeset', 'builddate',
                     'basis_type', 'poly_order', 'is_modal', 'lower',
                     'upper', 'cells', 'num_comps', 'grid_type',
                     'num_cdim', 'num_vdim']:
        output += '\n├─ {:s}: {}'.format(key, self.ctx[key])
      #end
    #end

    return output
  #end


  #---- Write ----------------------------------------------------------
  def write(self,
            out_name: str = None,
            mode: str = 'gkyl',
            var_name: str = None,
            bufferSize: int = 1000,
            append = False,
            cleaning = True):
    """Writes data in ADIOS .bp file, ASCII .txt file, or NumPy .npy file
    """
    # Create output file name
    if out_name is None:
      if self._file_name is not None:
        fn = self._file_name
        out_name = fn.split('.')[0].strip('_') + '_mod.' + mode
      else:
        out_name = 'gdata.' + mode
      #end
    else:
      if not isinstance(out_name, str):
        raise TypeError('\'out_name\' must be a string')
      #end
      if out_name.split('.')[-1] != mode:
        out_name += '.' + mode
      #end
    #end

    num_dims = self.get_num_dims()
    num_comps = self.get_num_comps()
    num_cells = self.get_num_cells()
    lo, up = self.get_bounds()
    values = self.get_values()

    full_shape = list(num_cells) + [num_comps]
    offset = [0] * (num_dims + 1)

    if var_name is None:
      var_name = self._var_name
    #end

    values = np.empty_like(self.get_values())
    values[...] = self.get_values()

    if mode == 'bp':
      import adios2
      if not append:
        fh = adios2.open(out_name, "w", engine_type="BP3")
        fh.write_attribute('numCells', num_cells)
        fh.write_attribute('lowerBounds', lo)
        fh.write_attribute('upperBounds', up)

        if self.ctx['time']:
          fh.write('time', self.ctx['time'])
        #end
      else:
        fh = adios2.open(out_name, "a", engine_type="BP3")
      #end
      fh.write(var_name, values, full_shape, offset, full_shape)
      fh.close()

      # Cleaning
      if cleaning:
        if len(out_name.split('/')) > 1:
          nm = out_name.split('/')[-1]
        else:
          nm = out_name
        #end
        shutil.move(out_name + '.dir/' + nm + '.0', out_name)
        shutil.rmtree(out_name + '.dir')
      #end
    elif mode == 'gkyl':
      dti = np.dtype('i8')
      dtf = np.dtype('f8')

      fh = open(out_name, 'w')

      # sep='' results in a binary file
      np.array([103, 107, 121, 108, 48],
               dtype=np.dtype('b')).tofile(fh, sep='')
      # version 1
      np.array([1], dtype=dti).tofile(fh, sep='')
      # type 1
      np.array([1], dtype=dti).tofile(fh, sep='')
      # meta size
      np.array([0], dtype=dti).tofile(fh, sep='')
      # real type (double)
      np.array([2], dtype=dti).tofile(fh, sep='')
      # num dims
      np.array([num_dims], dtype=dti).tofile(fh, sep='')
      # num cells
      np.array(num_cells, dtype=dti).tofile(fh, sep='')
      # lower
      np.array(lo, dtype=dtf).tofile(fh, sep='')
      # upper
      np.array(up, dtype=dtf).tofile(fh, sep='')
      # elem_sz
      np.array([num_comps*8], dtype=dti).tofile(fh, sep='')
      # asize
      np.array([len(values)], dtype=dti).tofile(fh, sep='')
      # data
      np.array(values, dtype=dtf).tofile(fh, sep='')

      fh.close()
    elif mode == 'txt':
      numRows = int(num_cells.prod())
      grid = self.get_grid()
      for d in range(num_dims):
        grid[d] = 0.5*(grid[d][1:]+grid[d][:-1])
      #end

      basis = np.full(num_dims, 1.0)
      for d in range(num_dims-1):
        basis[d] = num_cells[(d+1):].prod()
      #end

      fh = open(out_name, 'w')
      for i in range(numRows):
        idx = i
        idxs = np.zeros(num_dims, np.int32)
        for d in range(num_dims):
          idxs[d] = int(idx // basis[d])
          idx = idx % basis[d]
        #end
        line = ''
        for d in range(num_dims):
          line += '{:.15e}, '.format(grid[d][idxs[d]])
        #end
        for c in range(num_comps-1):
          line += '{:.15e}, '.format(values[tuple(idxs)][c])
        #end
        line += '{:.15e}\n'.format(values[tuple(idxs)][num_comps-1])
        fh.write(line)
      #end
      fh.close()
    elif mode == 'npy':
      np.save(out_name, values.squeeze())
    #end
  #end
#end
