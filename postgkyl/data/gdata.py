import glob
import os.path
import numpy as np
import shutil
from typing import Union

from postgkyl.data.load_gkyl import load_gkyl
from postgkyl.data.load_h5 import load_h5
from postgkyl.data.load_flash import load_flash
from postgkyl.utils import idxParser

class GData(object):
  """Provides interface to Gkeyll output data.

  Data serves as a baseline interface to Gkeyll data. It is used for
  loading Gkeyll data and serves is input to many Postgkyl
  functions. Represents a dataset in the Postgkyl command line mode.

  Attributes:
    meta: A dictionary contaioning a physics information like charge
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
               meta: dict = None,
               comp_grid: bool = False,
               mapc2p_name: str = None,
               source : str = None) -> None:
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
      meta: dict
        Copy content of the specified meta dictionary.
      comp_grid: bool
        A flag to ignore grid mapping.
      mapc2p_name: str
        The name of the file containg the c2p mapping information.
    """
    self._tag = tag
    self._comp_grid = comp_grid # disregard the mapped grid
    self._grid = None
    self._gridType = 'uniform'
    self._gridFile = None
    self._values = None # (N+1)D narray of values

    self.meta = {}
    self.meta['time'] = None
    self.meta['frame'] = None
    self.meta['changeset'] = None
    self.meta['builddate'] = None
    self.meta['polyOrder'] = None
    self.meta['basisType'] = None
    self.meta['isModal'] = None
    if meta:
      for key in meta:
        self.meta[key] = meta[key]
      #end
    #end

    self._label = ''
    self._customLabel = label
    self._var_name = var_name
    self.file_name = file_name
    self.mapc2p_name = mapc2p_name
    if file_name is not None:
      # Sequence load typically concatenates multiple files
      # When the sequence is in just a single file, _loadFrame will
      # fail and _loadSequence is called instead
      if os.path.isfile(self.file_name):
        zs = (z0, z1, z2, z3, z4, z5)
        self._loadFrame(axes = zs, comp = comp,
                        source = source)
      else:
        self._loadSequence()
      #end
    #end

    self.color = None

    self._status = True
    self._source = source
  #end

  #---- File Loading ---------------------------------------------------
  def _createOffsetCountBp(self, bpVar, zs, comp, grid=None):
    num_dims = len(bpVar.dims)
    count = np.array(bpVar.dims)
    offset = np.zeros(num_dims, np.int32)
    cnt = 0
    for d, z in enumerate(zs):
      if d < num_dims-1 and z is not None:  # Last dim stores comp
        z = idxParser(z, grid[d])
        if isinstance(z, int):
          offset[d] = z
          count[d] = 1
        elif isinstance(z, slice):
          offset[d] = z.start
          count[d] = z.stop - z.start
        else:
          raise TypeError('\'z\' is neither number or slice')
        #end
        cnt = cnt + 1
      #end
    #end

    if comp is not None:
      comp = idxParser(comp)
      if isinstance(comp, int):
        offset[-1] = comp
        count[-1] = 1
      elif isinstance(comp, slice):
        offset[-1] = comp.start
        count[-1] = comp.stop - comp.start
      else:
        raise TypeError('\'comp\' is neither number or slice')
      #end
      cnt = cnt + 1
    #end

    if cnt > 0:
      return tuple(offset), tuple(count)
    else:
      return (), ()
    #end
  #end

  def _loadFrame(self,
                 axes : tuple = (None, None, None, None, None, None),
                 comp : int = None,
                 source : str = None):
    self._file_dir = os.path.dirname(os.path.realpath(self.file_name))
    extension = self.file_name.split('.')[-1]
    if extension == 'h5':
      status, params = load_h5(self.file_name, self.meta)
      if status:
        lower = params[0]
        upper = params[1]
        cells = params[2]
        self._values = params[3]
      else:
        self._loadSequence()
        return
      #end
    elif extension == 'bp':
      import adios
      fh = adios.file(self.file_name)
      if not self._var_name in fh.vars:
        # Not a Gkyl 'frame' data; trying to load as a sequence
        fh.close()
        self._loadSequence()
        return
      #end
      # Get the atributes
      self.attrsList = { }
      for k in fh.attrs.keys():
        self.attrsList[k] = 0
      #end
      # Postgkyl conventions require the attributes to be
      # narrays even for 1D data
      lower = np.atleast_1d(adios.attr(fh, 'lowerBounds').value)
      upper = np.atleast_1d(adios.attr(fh, 'upperBounds').value)
      cells = np.atleast_1d(adios.attr(fh, 'numCells').value)
      if 'changeset' in fh.attrs.keys():
        self.meta['changeset'] = adios.attr(fh, 'changeset').value.decode('UTF-8')
      #end
      if 'builddate' in fh.attrs.keys():
        self.meta['builddate'] = adios.attr(fh, 'builddate').value.decode('UTF-8')
      #end
      if 'polyOrder' in fh.attrs.keys():
        self.meta['polyOrder'] = adios.attr(fh, 'polyOrder').value
        self.meta['isModal'] = True
      #end
      if 'basisType' in fh.attrs.keys():
        self.meta['basisType'] = adios.attr(fh, 'basisType').value.decode('UTF-8')
        self.meta['isModal'] = True
      #end
      if 'charge' in fh.attrs.keys():
        self.meta['charge'] = adios.attr(fh, 'charge').value
      #end
      if 'mass' in fh.attrs.keys():
        self.meta['mass'] = adios.attr(fh, 'mass').value
      #end
      if 'time' in fh.vars:
        self.meta['time'] = adios.var(fh, 'time').read()
      #end
      if 'frame' in fh.vars:
        self.meta['frame'] = adios.var(fh, 'frame').read()
      #end

      # Check for mapped grid ...
      if 'type' in fh.attrs.keys() and self._comp_grid is False:
        self._gridType = adios.attr(fh, 'type').value.decode('UTF-8')
      #end
      # .. load nodal grid if provided ...
      if self._gridType == 'uniform':
        pass # nothing to do for uniform grids
      elif self._gridType == 'mapped':
        if 'grid' in fh.attrs.keys():
          gridNm = self._file_dir + '/' +adios.attr(fh, 'grid').value.decode('UTF-8')
        else:
          gridNm = self._file_dir + '/grid'
        #end
        with adios.file(gridNm) as gridFh:
          gridVar = adios.var(gridFh, self._var_name)
          offset, count = self._createOffsetCountBp(gridVar, axes, None)
          tmp = gridVar.read(offset=offset, count=count)
          grid = [tmp[..., d].transpose()
                  #for d in range(len(cells))]
                  for d in range(tmp.shape[-1])]
          self._grid = grid
        #end
      elif self._gridType == 'nonuniform':
        raise TypeError('\'nonuniform\' is not presently supported')
      else:
        raise TypeError('Unsupported grid type info in field!')
      #end

      # Load data
      num_dims = len(cells)
      var = adios.var(fh, self._var_name)
      grid = [np.linspace(lower[d],
                          upper[d],
                          cells[d]+1)
              for d in range(num_dims)]
      offset, count = self._createOffsetCountBp(var, axes, comp, grid)
      self._values = var.read(offset=offset, count=count, nsteps=1)

      # Adjust boundaries for 'offset' and 'count'
      dz = (upper - lower) / cells
      if offset:
        if self._gridType == 'uniform':
          lower = lower + offset[:num_dims]*dz
          cells = cells - offset[:num_dims]
        elif self._gridType == 'mapped':
          idx = np.full(num_dims, 0)
          for d in range(num_dims):
            lower[d] = self._grid[d][tuple(idx)]
            cells[d] = cells[d] - offset[d]
          #end
        #end
      #end
      if count:
        if self._gridType == 'uniform':
          upper = lower + count[:num_dims]*dz
          cells = count[:num_dims]
        elif self._gridType == 'mapped':
          idx = np.full(num_dims, 0)
          for d in range(num_dims):
            idx[-d-1] = count[d]-1  #.Reverse indexing of idx because of transpose() in composing self._grid.
            upper[d] = self._grid[d][tuple(idx)]
            cells[d] = count[d]
          #end
        #end
      #end
      fh.close()
    elif extension == 'gkyl':
      num_dims, cells, grid, values = load_gkyl(self.file_name)
      if isinstance(grid, tuple):
        lower = grid[0]
        upper = grid[1]
      else:
        lower = np.array([grid[0]])
        upper = np.array([grid[-1]])
        self._grid = [grid]
      #end
      self._values = values
    elif source == 'flash':
      num_dims, cells, extends, values = load_flash(self.file_name,
                                                    self._var_name)
      lower = extends[0]
      upper = extends[1]
      #end
      self._values = values[..., np.newaxis]
    else:
      raise NameError(
        'File extension \'{:s}\' is not supported'.format(extension))
      #end
    #end

    if self.mapc2p_name is not None:
      extension = self.mapc2p_name.split('.')[-1]
      self._gridType = 'c2p'
      if extension == 'gkyl':
        num_dims, _, _, grid = load_gkyl(self.mapc2p_name)
        num_comps = grid.shape[-1]
        num_coeff = num_comps/num_dims
        self._grid = [grid[..., int(d*num_coeff):int((d+1)*num_coeff)]
                      for d in range(num_dims)]
      else:
        raise NameError(
          'File extension \'{:s}\' is not supported for mapc2p'.format(extension))
      #end
    #end

    num_dims = len(cells)
    dz = (upper - lower) / cells
    # Adjusts bounds in case ghost layer is included in data
    for d in range(num_dims):
      if cells[d] != self._values.shape[d]:
        if self._gridType == 'mapped':
          raise ValueError(
            'Data appears to include ghost cells which is not compatible with mapped grid.' \
            'Use computational grid \'comp_grid\' instead.')
        #end
        ngl = int(np.floor((cells[d] - self._values.shape[d])*0.5))
        ngu = int(np.ceil((cells[d] - self._values.shape[d])*0.5))
        cells[d] = self._values.shape[d]
        lower[d] = lower[d] - ngl*dz[d]
        upper[d] = upper[d] + ngu*dz[d]
      #end
    #end

    # Construct grids if not loaded already
    if self._grid is None:
      self._grid = [np.linspace(lower[d],
                                upper[d],
                                cells[d]+1)
                    for d in range(num_dims)]
    #end
  #end

  def _loadSequence(self):
    # Sequence load typically cancatenates multiple files
    files = glob.glob('{:s}*'.format(self.file_name))
    if not files:
      raise NameError(
        'File(s) \'{:s}\' not found or empty.'.
        format(self.file_name))

    cnt = 0  # Counter for the number of loaded files
    for file_name in files:
      extension = file_name.split('.')[-1]
      if extension == 'h5':
        import tables
        self.fileType = 'hdf5'
        fh = tables.open_file(file_name, 'r')
        if '/DataStruct/data' in fh and \
           '/DataStruct/timeMesh' in fh:
          grid = fh.root.DataStruct.timeMesh.read()
          values = fh.root.DataStruct.data.read()
          fh.close()
        else:
          fh.close()
          continue
        #end
      elif extension == 'bp':
        import adios
        self.fileType = 'adios'
        fh = adios.file(file_name)
        timeMeshList = [key for key, _ in fh.vars.items() if 'TimeMesh' in key]
        dataList = [key for key, _ in fh.vars.items() if 'Data' in key]
        if len(dataList) > 0:
          for i in range(len(dataList)):
            if i==0:
              values = adios.var(fh, dataList[i]).read()
              grid = adios.var(fh, timeMeshList[i]).read()
            else:
              newvals = np.asarray(adios.var(fh, dataList[i]).read())
              newgrid = np.asarray(adios.var(fh, timeMeshList[i]).read())
              # deal with weird behavior after restart where some data is a scalar
              if len(newvals.shape) == 0:
                  newvals = np.reshape(newvals, (1,1))
                  newgrid = np.reshape(newgrid, (1,))
              # deal with weird behavior after restart where some data
              # doesn't have second dimension
              if len(newvals.shape) < 2:
                newvals = np.expand_dims(newvals, axis=1)
              #end
              values = np.append(values, newvals,axis=0)
              grid = np.append(grid, newgrid,axis=0)
            #end
          #end
          fh.close()
        else:
          fh.close()
          continue
        #end
      else:
        continue
      #end

      if cnt > 0:
        self._grid = np.append(self._grid, grid, axis=0)
        self._values = np.append(self._values, values, axis=0)
      else:
        self._grid = grid
        self._values = values
      #end
      cnt += 1
    #end

    if cnt == 0:  # No files loaded
      raise NameError(
        'File(s) \'{:s}\' not found or empty.'.
        format(self.file_name))
    #end
    # Squeeze the time coordinate ...
    if len(self._grid.shape) > 1:
      self._grid = np.squeeze(self._grid)
    #end
    # ... and make it a list following the Postgkyl grid conventions
    self._grid = [self._grid]

    # glob() doesn't guarantee the right order
    sortIdx = np.argsort(self._grid[0])
    self._grid[0] = self._grid[0][sortIdx]
    self._values = self._values[sortIdx, ...]
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
    import adios
    fh = adios.file(self.file_name)
    inputFile = adios.attr(fh, 'inputfile').value.decode('UTF-8')
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
    return self._gridType
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

      if self.meta['time'] is not None:
        output += '├─ Time: {:e}\n'.format(self.meta['time'])
      #end
      if self.meta['frame'] is not None:
        output += '├─ Frame: {:d}\n'.format(self.meta['frame'])
      #end
      output += '├─ Number of components: {:d}\n'.format(numComps)
      output += '├─ Number of dimensions: {:d}\n'.format(num_dims)
      output += '├─ Grid: ({:s})\n'.format(self._gridType)
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
      if self.meta['polyOrder'] and self.meta['basisType']:
        output += '\n├─ DG info:\n'
        output += '│  ├─ Polynomial Order: {:d}\n'.format(self.meta['polyOrder'])
        if self.meta['isModal']:
          output += '│  └─ Basis Type: {:s} (modal)'.format(self.meta['basisType'])
        else:
          output += '│  └─ Basis Type: {:s}'.format(self.meta['basisType'])
        #end
      #end
      if self.meta['changeset'] and self.meta['builddate']:
        output += '\n├─ Created with Gkeyll:\n'
        output += '│  ├─ Changeset: {:s}\n'.format(self.meta['changeset'])
        output += '│  └─ Build Date: {:s}'.format(self.meta['builddate'])
      #end
      for key in self.meta:
        if key not in ['time', 'frame', 'changeset', 'builddate',
                       'basisType', 'polyOrder', 'isModal']:
          output += '\n├─ {:s}: {}'.format(key, self.meta[key])
        #end
      #end

      #output += '\n- Contains attributes:\n  '
      #for k in self.attrsList:
      #    output += '{:s} '.format(k)
      #end

      return output
    else:
      return 'No data'
    #end
  #end


  #---- Write ----------------------------------------------------------
  def write(self,
            out_name: str = None,
            mode: str = 'bp',
            var_name: str = None,
            bufferSize: int = 1000,
            append = False,
            cleaning = True):
    """Writes data in ADIOS .bp file, ASCII .txt file, or NumPy .npy file
    """
    import adios
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
    numComps = self.getNumComps()
    numCells = self.getNumCells()

    if mode == 'bp':
      # Create string number of cells and offsets
      sNumCells = ''
      sOffsets = ''
      for i in range(num_dims):
        sNumCells += '{:d},'.format(int(numCells[i]))
        sOffsets += '0,'
      #end
      sNumCells += '{:d}'.format(numComps)
      sOffsets += '0'

      if var_name is None:
        var_name = self._var_name
      #end

      if not append:
        adios.init_noxml()
        adios.set_max_buffer_size(bufferSize)
        groupId = adios.declare_group('CartField', '')
        adios.select_method(groupId, 'POSIX1', '', '')

        # Define variables and attributes
        adios.define_attribute_byvalue(groupId, 'numCells', '', numCells)
        lo, up = self.getBounds()
        adios.define_attribute_byvalue(groupId, 'lowerBounds', '', lo)
        adios.define_attribute_byvalue(groupId, 'upperBounds', '', up)
        fh = adios.open('CartField', out_name, 'w')

        if self.meta['time']:
          adios.define_var(groupId, 'time', '',
                           adios.DATATYPE.double, '', '', '')
          adios.write(fh, 'time', self.meta['time'])
        #end

        adios.define_var(groupId, var_name, '',
                         adios.DATATYPE.double,
                         sNumCells, sNumCells, sOffsets)
        adios.write(fh, var_name, self.getValues())

        adios.close(fh)
        adios.finalize()
      else:
        adios.init_noxml()
        adios.set_max_buffer_size(bufferSize)
        groupId = adios.declare_group('CartField', '')
        adios.select_method(groupId, 'POSIX1', '', '')

        fh = adios.open('CartField', out_name, 'a')

        adios.define_var(groupId, var_name, '',
                         adios.DATATYPE.double,
                         sNumCells, sNumCells, sOffsets)
        adios.write(fh, var_name, self.getValues())
        adios.close(fh)
        adios.finalize()
      #end

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
    elif mode == 'txt':
      numRows = int(numCells.prod())
      grid = self.getGrid()
      for d in range(num_dims):
        grid[d] = 0.5*(grid[d][1:]+grid[d][:-1])
      #end
      values = self.getValues()

      basis = np.full(num_dims, 1.0)
      for d in range(num_dims-1):
        basis[d] = numCells[(d+1):].prod()
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
        for c in range(numComps-1):
          line += '{:.15e}, '.format(values[tuple(idxs)][c])
        #end
        line += '{:.15e}\n'.format(values[tuple(idxs)][numComps-1])
        fh.write(line)
      #end
      fh.close()
    elif mode == 'npy':
      values = self.getValues()
      np.save(out_name, values.squeeze())
    #end
  #end
#end
