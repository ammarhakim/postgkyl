import glob
import os.path
import numpy as np
import shutil
from typing import Union

from postgkyl.utils import idxParser


class Data(object):
  """Provides interface to Gkeyll output data.

  Data serves as a baseline interface to Gkeyll data. It allows for
  loading and saving and serves as an input for other Postgkyl
  methods. Represents a dataset in the Postgkyl command line mode.

  Init Args:
    fileName (str): String with either full file name of or a
      fragment (for history loading).  Currently supports only
      'h5' and 'bp' files. Empty GData object is created when
      not specified
    stack (bool): Turn the internal stack on (True) and off
      (False) (default: True)
    comp (int or 'int:int'): Preselect a componend index (or a
      slice) for the partial load of data
    z0 (int or 'int:int'): Preselect an index (or a slice) for
      the first coordinate for the partial load of data
    z1 (int or 'int:int'): Preselect an index (or a slice) for
      the second coordinate for the partial load of data
    z2 (int or 'int:int'): Preselect an index (or a slice) for
      the third coordinate for the partial load of data
    z3 (int or 'int:int'): Preselect an index (or a slice) for
      the fourth coordinate for the partial load of data
    z4 (int or 'int:int'): Preselect an index (or a slice) for
      the fifth coordinate for the partial load of data
    z5 (int or 'int:int'): Preselect an index (or a slice) for
      the sixth coordinate for the partial load of data
    varName (str): allows to specify Adios variable; default is CartGridField

    Raises:
        NameError: when file name does not exist or is empty/corrupted.
        TypeError: when partial load indices are no integers or 'int:int'

    Notes:
        - When fine name is incomplete or is not a Gkyl frame,
          Postgkyl tries opening it as a history data before throwing
          an exception
        - Preselection coordinate indices for higher dimension than
          included in the data are safelly disregarded (i.e., z2,
          z3, z4, and z5 for 2D data).
        - postgkyl.Data is a shortcut for postgkyl.data.Data

    Examples:
        import postgkyl
        data = postgkyl.GData('file.bp', comp=1)
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
               compgrid: bool = False,
               varName: str = 'CartGridField',
               tag: str = 'default',
               label = None,
               meta = None,
               mapc2p_name: str = None,
               fv: bool = False) -> None:
    self._tag = tag
    self._compGrid = compgrid # disregard the mapped grid
    self._grid = None
    self._gridType = 'uniform'
    self._gridFile = None
    self._values = None # (N+1)D narray of values
    self._lower = None
    self._upper = None

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
    self._varName = varName
    self.file_name = file_name
    self.mapc2p_name = mapc2p_name
    self._fv = fv
    if file_name is not None:
      # Sequence load typically concatenates multiple files
      # When the sequence is in just a single file, _loadFrame will
      # fail and _loadSequence is called instead
      if os.path.isfile(self.file_name):
        zs = (z0, z1, z2, z3, z4, z5)
        self._loadFrame(zs, comp)
      else:
        self._loadSequence()
      #end
    #end
    
    self.color = None

    self._status = True
  #end

  #-----------------------------------------------------------------
  #-- File Loading -------------------------------------------------
  def _createOffsetCountBp(self, bpVar, zs, comp):
    num_dims = len(bpVar.dims)
    count = np.array(bpVar.dims)
    offset = np.zeros(num_dims, np.int)
    cnt = 0

    for d, z in enumerate(zs):
      if d < num_dims-1 and z is not None:  # Last dim stores comp
        z = idxParser(z)
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
        offset[d] = comp.start
        count[d] = comp.stop - comp.start
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

  def _load_gkyl(self, file_name) -> tuple:
    dti8 = np.dtype('i8')
    dtf = np.dtype('f8')
    doffset = 8

    offset = 0

    # read real-type
    realType = np.fromfile(file_name, dtype=dti8, count=1)[0]
    if realType == 1:
      dtf = np.dtype('f4')
      doffset = 4
    #end

    offset += 8
            
    # read grid dimensions
    num_dims = np.fromfile(file_name, dtype=dti8, count=1, offset=offset)[0]
    offset += 8

    # read grid shape
    cells = np.fromfile(file_name, dtype=dti8, count=num_dims, offset=offset)
    offset += num_dims*8

    # read lower/upper
    lower = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
    offset += num_dims*doffset

    upper = np.fromfile(file_name, dtype=dtf, count=num_dims, offset=offset)
    offset += num_dims*doffset

    # read array elemEz (the div by doffset is as elemSz includes sizeof(real_type) = doffset)
    elemSzRaw = int(np.fromfile(file_name, dtype=dti8, count=1, offset=offset)[0])
    elemSz = elemSzRaw/doffset
    offset += 8

    # read array size
    asize = np.fromfile(file_name, dtype=dti8, count=1, offset=offset)[0]
    offset += 8

    adata = np.fromfile(file_name, dtype=dtf, offset=offset)
    gshape = np.ones(num_dims+1, dtype=np.dtype('i8'))
    for d in range(num_dims):
      gshape[d] = cells[d]
    #end
    numComp = elemSz
    gshape[-1] = int(numComp)
    return num_dims, cells, lower, upper, adata.reshape(gshape)
  #end

  def _loadFrame(self, axes=(None, None, None, None, None, None),
                 comp=None):
    self.fileDir = os.path.dirname(os.path.realpath(self.file_name))
    extension = self.file_name.split('.')[-1]
    if extension == 'h5':
      import tables
      fh = tables.open_file(self.file_name, 'r')
      if not '/StructGridField' in fh:
        fh.close()
        self._loadSequence()
        return
      #end
            
      # Get the atributes
      lower = fh.root.StructGrid._v_attrs.vsLowerBounds
      upper = fh.root.StructGrid._v_attrs.vsUpperBounds
      cells = fh.root.StructGrid._v_attrs.vsNumCells
      # Load data ...
      self._values = fh.root.StructGridField.read()
      # ... and the time-stamp
      if '/timeData' in fh:
        self.meta['time'] = fh.root.timeData._v_attrs.vsTime
      #end
      fh.close()
    elif extension == 'bp':
      import adios
      fh = adios.file(self.file_name)
      if not self._varName in fh.vars:
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
      if 'type' in fh.attrs.keys() and self._compGrid is False:
        self._gridType = adios.attr(fh, 'type').value.decode('UTF-8')
      #end
      # .. load nodal grid if provided ...
      if self._gridType == 'uniform':
        pass # nothing to do for uniform grids
      elif self._gridType == 'mapped':
        if 'grid' in fh.attrs.keys():
          gridNm = self.fileDir + '/' +adios.attr(fh, 'grid').value.decode('UTF-8')
        else:
          gridNm = self.fileDir + '/grid'  
        #end
        with adios.file(gridNm) as gridFh:
          gridVar = adios.var(gridFh, self._varName)
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
      var = adios.var(fh, self._varName)
      offset, count = self._createOffsetCountBp(var, axes, comp)
      self._values = var.read(offset=offset, count=count)

      # Adjust boundaries for 'offset' and 'count'
      num_dims = len(cells)
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
      num_dims, cells, lower, upper, values = self._load_gkyl(self.file_name)
      self._values = values
    else:
      raise NameError(
        'File extension \'{:s}\' is not supported'.format(extension))
    #end

    self._upper = upper
    self._lower = lower
    
    if self.mapc2p_name is not None:
      extension = self.mapc2p_name.split('.')[-1]
      self._gridType = 'c2p'
      if extension == 'gkyl':
        num_dims, _, _, _, grid = self._load_gkyl(self.mapc2p_name)
        num_comps = grid.shape[-1]
        num_coeff = int(num_comps/num_dims)
        self._grid = [grid[..., int(d*num_coeff):int((d+1)*num_coeff)]
                      for d in range(num_dims)]
        if self._fv:
          for d in range(num_dims):
            self._grid[d] = self._grid[d][..., 0]/2**(num_dims/2)
          #end
        #end
        print(self._grid[1])
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
            'Use computational grid \'compgrid\' instead.')
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
      self._grid = [np.linspace(lower[d]+dz[d]/2,
                                upper[d]-dz[d]/2,
                                cells[d])
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
        fh =  adios.file(file_name)
        timeMeshList = [key for key, val in fh.vars.items() if 'TimeMesh' in key]
        dataList = [key for key, val in fh.vars.items() if 'Data' in key]
        if len(dataList) > 0:
          for i in range(len(dataList)):
            if i==0:
              values = adios.var(fh, dataList[i]).read()
              grid = adios.var(fh, timeMeshList[i]).read()
            else:
              newvals = adios.var(fh, dataList[i]).read()
              # deal with weird behavior after restart where some data
              # doesn't have second dimension
              if len(newvals.shape) < 2:
                newvals = np.expand_dims(newvals, axis=1)
              #end
              values = np.append(values, newvals,axis=0)
              grid = np.append(grid,
                               adios.var(fh, timeMeshList[i]).read(),
                               axis=0)
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


  #-----------------------------------------------------------------
  #-- Stuff Control ------------------------------------------------
  def getTag(self):
    return self._tag
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
    if self._lower is not None and self._upper is not None:
      return self._lower, self._upper
    elif self._grid is not None:
      num_dims = len(self._values.shape)-1
      lo, up, dz = np.zeros(num_dims), np.zeros(num_dims), np.zeros(num_dims)
      for d in range(num_dims):
        lo[d] = self._grid[d].min()
        up[d] = self._grid[d].max()
        dz[d] = (self._grid[d].max()-self._grid[d].min())/len(self._grid[d])
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

  def setGrid(self, grid,
              lower = None, upper = None):
    self._grid = grid
    if lower is not None:
      self._lower = lower
    #end
    if upper is not None:
      self._upper = upper
    #end
  #end

  def setValues(self, values):
    self._values = values
  #end                       

  def push(self, grid, values,
           lower = None, upper = None):
    self.setValues(values)
    self.setGrid(grid, lower, upper)
    return self
  #end

  #-----------------------------------------------------------------
  #-- Info ---------------------------------------------------------
  def info(self):
    """Prints GData object information.
    
    Prints time (only when available), number of components, dimension
    spans, extremes for a GData object. Only the top of the stack is
    printed.
        
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
        output += '│  ├─ Dim {:d}: Num. cells: {:d}; '.format(
          d, numCells[d])
        output += 'Lower: {:e}; Upper: {:e}\n'.format(
          lower[d], upper[d])
        #end
      output += '│  └─ Dim {:d}: Num. cells: {:d}; '.format(
        num_dims-1, numCells[num_dims-1])
      output += 'Lower: {:e}; Upper: {:e}\n'.format(
        lower[num_dims-1], upper[num_dims-1])
      maximum = np.nanmax(values)
      maxIdx = np.unravel_index(np.nanargmax(values), values.shape)
      minimum = np.nanmin(values)
      minIdx = np.unravel_index(np.nanargmin(values), values.shape)
      output += '├─ Maximum: {:e} at {:s}'.format(
        maximum, str(maxIdx[:num_dims]))
      if numComps > 1:
        output += ' component {:d}\n'.format(maxIdx[-1])
      else:
        output += '\n'
      #end
      output += '├─ Minimum: {:e} at {:s}'.format(
        minimum, str(minIdx[:num_dims]))
      if numComps > 1:
        output += ' component {:d}'.format(minIdx[-1])
      #end
      if self.meta['polyOrder'] is not None \
         and self.meta['basisType'] is not None:
        output += '\n├─ DG info:\n'
        output += '│  ├─ Polynomial Order: {:d}\n'.format(
          self.meta['polyOrder'])
        if self.meta['isModal']:
          output += '│  └─ Basis Type: {:s} (modal)'.format(
            self.meta['basisType'])
        else:
          output += '│  └─ Basis Type: {:s}'.format(
            self.meta['basisType'])
        #end
      #end
      if self.meta['changeset'] and self.meta['builddate']:
        output += '\n├─ Created with Gkeyll:\n'
        output += '│  ├─ Changeset: {:s}\n'.format(self.meta['changeset'])
        output += '│  └─ Build Date: {:s}'.format(self.meta['builddate'])
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


  #-----------------------------------------------------------------
  #-- Write --------------------------------------------------------
  def write(self, bufferSize=1000, outName=None, mode='bp'):
    """Writes data in ADIOS .bp file, ASCII .txt file, or NumPy .npy file
    """
    # Create output file name
    if outName is None:
      if self.file_name is not None:
        fn = self.file_name
        outName = fn.split('.')[0].strip('_') + '_mod.' + mode
      else:
        outName = 'gdata.' + mode
      #end
    else:
      if not isinstance(outName, str):
        raise TypeError('\'outName\' must be a string')
      #end
      if outName.split('.')[-1] != mode:
        outName += '.' + mode
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

      # ADIOS init
      adios.init_noxml()
      adios.set_max_buffer_size(bufferSize)
      groupId = adios.declare_group('CartField', '')
      adios.select_method(groupId, 'POSIX1', '', '')

      # Define variables and attributes
      adios.define_attribute_byvalue(groupId, 'numCells', '', numCells)
      lo, up = self.getBounds()
      adios.define_attribute_byvalue(groupId, 'lowerBounds', '', lo)
      adios.define_attribute_byvalue(groupId, 'upperBounds', '', up)
      fh = adios.open('CartField', outName, 'w')
            
      if self.meta['time']:
        adios.define_var(groupId, 'time', '',
                         adios.DATATYPE.double, '', '', '')
        adios.write(fh, key, self.meta['time'])
      #end
            
      adios.define_var(groupId, self._varName, '',
                       adios.DATATYPE.double,
                       sNumCells, sNumCells, sOffsets)

      adios.write(fh, self._varName, self.getValues())
      adios.close(fh)
      adios.finalize()

      # Cheating
      if len(outName.split('/')) > 1:
        nm = outName.split('/')[-1]
      else:
        nm = outName
      #end
      shutil.move(outName + '.dir/' + nm + '.0', outName)
      shutil.rmtree(outName + '.dir')
    elif mode == 'txt':
      numRows = int(numCells.prod())
      grid = self.getGrid()
      values = self.getValues()

      basis = np.full(num_dims, 1.0)
      for d in range(num_dims-1):
        basis[d] = numCells[(d+1):].prod()
      #end

      fh = open(outName, 'w')
      for i in range(numRows):
        idx = i
        idxs = np.zeros(num_dims, np.int)
        for d in range(num_dims):
          idxs[d] = int(idx // basis[d])
          idx = idx % basis[d]
        #end
        line = ''
        for d in range(num_dims-1):
          line += '{:.15e}, '.format(grid[d][idxs[d]])
        #end
        line += '{:.15e}, '.format(grid[num_dims-1][idxs[num_dims-1]])
        for c in range(numComps-1):
          line += '{:.15e}, '.format(values[tuple(idxs)][c])
        #end
        line += '{:.15e}\n'.format(values[tuple(idxs)][numComps-1])
        fh.write(line)
      #end
      fh.close()
    elif mode == 'npy':
      values = self.getValues()
      np.save(outName, values.squeeze())
    #end
  #end
#end
