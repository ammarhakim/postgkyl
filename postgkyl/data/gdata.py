import glob
import os.path
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
               file_name: str = None,
               comp: Union[int, str] = None,
               z0: Union[int, str] = None,
               z1: Union[int, str] = None,
               z2: Union[int, str] = None,
               z3: Union[int, str] = None,
               z4: Union[int, str] = None,
               z5: Union[int, str] = None,
               var_name: str = 'CartGridField',
               tag: str = 'default',
               label: str = None,
               ctx: dict = None,
               comp_grid: bool = False,
               mapc2p_name: str = None,
               reader_name: str = None) -> None:
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
    self._tag = tag
    self._comp_grid = comp_grid # disregard the mapped grid
    self._grid = None
    self._values = None # (N+1)D narray of values

    self.ctx = {}
    self.ctx['time'] = None
    self.ctx['frame'] = None
    self.ctx['changeset'] = None
    self.ctx['builddate'] = None
    self.ctx['polyOrder'] = None
    self.ctx['basisType'] = None
    self.ctx['isModal'] = None
    self.ctx['grid_type'] = 'uniform'
    if ctx:
      for key in ctx:
        self.ctx[key] = ctx[key]
      #end
    #end

    self._label = ''
    self._customLabel = label
    self._var_name = var_name
    file_name = str(file_name)
    self.file_name = file_name
    self.mapc2p_name = mapc2p_name
    self.color = None

    self._status = True

    zs = (z0, z1, z2, z3, z4, z5)

    self._readers = {
      'gkyl' : Read_gkyl,
      'adios' : Read_gkyl_adios,
      'h5' : Read_gkyl_h5,
      'flash' : Read_flash_h5
      }

    if file_name is not None:
      reader_set = False
      if reader_name in self._readers:
        self._reader = self._readers[reader_name](
          file_name=file_name,
          ctx=self.ctx,
          var_name=var_name,
          c2p=mapc2p_name,
          axes=zs, comp=comp)
        if self._reader._is_compatible():
          reader_set = True
        else:
          raise TypeError('{:s} cannot be read with the specified {:s} reader.'.format(self.file_name, reader_name))
        #end
      else:
        for key in self._readers:
          self._reader = self._readers[key](
            file_name=file_name,
            ctx= self.ctx,
            var_name=var_name,
            c2p=mapc2p_name,
            axes=zs, comp=comp)
          if self._reader._is_compatible():
            reader_set = True
            break
          #end
        #end
      #end
      if not reader_set:
        raise TypeError('"file_name" was specified ({:s}) but "reader" was either not set or successfully detected'.format(self.file_name))
      #end

      self._grid, self._values = self._reader.get_data()
    #end
  #end


  #---- Stuff Control --------------------------------------------------
  def getTag(self):
    return self._tag
  #end
  def setTag(self, tag=None):
    if tag:
      self._tag = tag
    #end
  #end

  def setLabel(self, label):
    self._label = label
  #end
  def getLabel(self):
    if self._customLabel:
      return self._customLabel
    else:
      return self._label
    #end
  #end
  def getCustomLabel(self):
    return self._customLabel
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

  def getInputFile(self):
    import adios2
    fh = adios2.open(self.file_name, 'rra')
    inputFile = fh.read_attribute_string('inputfile')[0]
    fh.close()
    return inputFile
  #end


  def getNumCells(self):
    if self._values is not None:
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

  def getNumComps(self):
    if self._values is not None:
      return int(self._values.shape[-1])
    else:
      return 0
    #end
  #end

  def getNumDims(self, squeeze=False):
    if self._values is not None:
      num_dims = int(len(self._values.shape)-1)
      if squeeze:
        cells = self.getNumCells()
        for d in range(num_dims):
          if cells[d] == 1:
            num_dims = num_dims - 1
          #end
        #end
      #end
      return num_dims
    else:
      return 0
    #end
  #end

  def getBounds(self):
    if self._grid is not None:
      num_dims = len(self._values.shape)-1
      lo, up = np.zeros(num_dims), np.zeros(num_dims)
      for d in range(num_dims):
        lo[d] = self._grid[d].min()
        up[d] = self._grid[d].max()
      #end
      return lo, up
    else:
      return np.array([]), np.array([])
    #end
  #end

  def getGrid(self):
    return self._grid
  #end

  def getGridType(self):
    return self.ctx['grid_type']
  #end

  def getValues(self):
    return self._values
  #end

  def setGrid(self, grid):
    self._grid = grid
  #end

  def setValues(self, values):
    self._values = values
  #end

  def push(self, grid, values):
    self._values = values
    self._grid = grid
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
    values = self.getValues()
    numComps = self.getNumComps()
    num_dims = self.getNumDims()
    numCells = self.getNumCells()
    lower, upper = self.getBounds()

    if len(values) > 0:
      output = ''

      if self.ctx['time'] is not None:
        output += '├─ Time: {:e}\n'.format(self.ctx['time'])
      #end
      if self.ctx['frame'] is not None:
        output += '├─ Frame: {:d}\n'.format(self.ctx['frame'])
      #end
      output += '├─ Number of components: {:d}\n'.format(numComps)
      output += '├─ Number of dimensions: {:d}\n'.format(num_dims)
      output += '├─ Grid: ({:s})\n'.format(self.getGridType())
      for d in range(num_dims-1):
        output += '│  ├─ Dim {:d}: Num. cells: {:d}; '.format(d, numCells[d])
        output += 'Lower: {:e}; Upper: {:e}\n'.format(lower[d],
                                                      upper[d])
        #end
      output += '│  └─ Dim {:d}: Num. cells: {:d}; '.format(num_dims-1, numCells[num_dims-1])
      output += 'Lower: {:e}; Upper: {:e}\n'.format(lower[num_dims-1],
                                                    upper[num_dims-1])
      maximum = np.nanmax(values)
      maxIdx = np.unravel_index(np.nanargmax(values), values.shape)
      minimum = np.nanmin(values)
      minIdx = np.unravel_index(np.nanargmin(values), values.shape)
      output += '├─ Maximum: {:e} at {:s}'.format(maximum,
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
      if self.ctx['polyOrder'] and self.ctx['basisType']:
        output += '\n├─ DG info:\n'
        output += '│  ├─ Polynomial Order: {:d}\n'.format(self.ctx['polyOrder'])
        if self.ctx['isModal']:
          output += '│  └─ Basis Type: {:s} (modal)'.format(self.ctx['basisType'])
        else:
          output += '│  └─ Basis Type: {:s}'.format(self.ctx['basisType'])
        #end
      #end
      if self.ctx['changeset'] and self.ctx['builddate']:
        output += '\n├─ Created with Gkeyll:\n'
        output += '│  ├─ Changeset: {:s}\n'.format(self.ctx['changeset'])
        output += '│  └─ Build Date: {:s}'.format(self.ctx['builddate'])
      #end
      for key in self.ctx:
        if key not in ['time', 'frame', 'changeset', 'builddate',
                       'basisType', 'polyOrder', 'isModal']:
          output += '\n├─ {:s}: {}'.format(key, self.ctx[key])
        #end
      #end

      return output
    else:
      return 'No data'
    #end
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
      if self.file_name is not None:
        fn = self.file_name
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

    num_dims = self.getNumDims()
    num_comps = self.getNumComps()
    num_cells = self.getNumCells()
    lo, up = self.getBounds()
    values = self.getValues()

    full_shape = list(num_cells) + [num_comps]
    offset = [0] * (num_dims + 1)

    if var_name is None:
      var_name = self._var_name
    #end

    values = np.empty_like(self.getValues())
    values[...] = self.getValues()

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

      np.array([103, 107, 121, 108, 48], dtype=np.dtype('b')).tofile(fh, sep='')  # sep='' results in a binary file
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
      grid = self.getGrid()
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
