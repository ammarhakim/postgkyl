import numpy as np
import os.path

from postgkyl.utils import idxParser

class Read_gkyl_adios(object):
  """Provides a framework to read gkyl Adios output
  """

  def __init__(self,
               file_name : str,
               ctx : dict = None,
               var_name : str = 'CartGridField',
               axes : tuple = (None, None, None, None, None, None),
               comp : int = None,
               **kwargs) -> None:
    self.file_name = file_name
    self.var_name = var_name

    self.axes = axes
    self.comp = comp

    self.is_frame = False
    self.is_diagnostic = False

    self.ctx = ctx
  #end

  def _is_compatible(self) -> bool:
    # Adios has been a problematic dependency; therefore it is only
    # imported when actially needed
    try:
      import adios
      fh = adios.file(self.file_name)

      for key, _ in fh.vars.items():
        if 'TimeMesh' in key:
          self.is_diagnostic = True
          break
        #end
        if self.var_name in key:
          self.is_frame = True
          break
        #end
      #end
      fh.close()
    except:
      return False
    return self.is_frame or self.is_diagnostic
  #end

  def _create_offset_count(self, var, zs, comp, grid=None) -> tuple:
    num_dims = len(var.dims)
    count = np.array(var.dims)
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

  def _read_frame(self) -> tuple:
    import adios
    fh = adios.file(self.file_name)

    # Postgkyl conventions require the attributes to be
    # narrays even for 1D data
    lower = np.atleast_1d(adios.attr(fh, 'lowerBounds').value)
    upper = np.atleast_1d(adios.attr(fh, 'upperBounds').value)
    cells = np.atleast_1d(adios.attr(fh, 'numCells').value)
    if 'changeset' in fh.attrs.keys():
      self.ctx['changeset'] = adios.attr(fh, 'changeset').value.decode('UTF-8')
    #end
    if 'builddate' in fh.attrs.keys():
      self.ctx['builddate'] = adios.attr(fh, 'builddate').value.decode('UTF-8')
    #end
    if 'polyOrder' in fh.attrs.keys():
      self.ctx['polyOrder'] = adios.attr(fh, 'polyOrder').value
      self.ctx['isModal'] = True
    #end
    if 'basisType' in fh.attrs.keys():
      self.ctx['basisType'] = adios.attr(fh, 'basisType').value.decode('UTF-8')
      self.ctx['isModal'] = True
    #end
    if 'charge' in fh.attrs.keys():
      self.ctx['charge'] = adios.attr(fh, 'charge').value
    #end
    if 'mass' in fh.attrs.keys():
      self.ctx['mass'] = adios.attr(fh, 'mass').value
    #end
    if 'time' in fh.vars:
      self.ctx['time'] = adios.var(fh, 'time').read()
    #end
    if 'frame' in fh.vars:
      self.ctx['frame'] = adios.var(fh, 'frame').read()
    #end

    # Check for mapped grid ...
    if 'type' in fh.attrs.keys():# and self._comp_grid is False:
      self.ctx['grid_type'] = adios.attr(fh, 'type').value.decode('UTF-8')
    #end
    # .. load nodal grid if provided ...
    # if self.ctx['grid_type']  == 'uniform':
    #   pass # nothing to do for uniform grids
    # elif self._gridType == 'mapped':
    #     if 'grid' in fh.attrs.keys():
    #       gridNm = self._file_dir + '/' +adios.attr(fh, 'grid').value.decode('UTF-8')
    #     else:
    #       gridNm = self._file_dir + '/grid'
    #     #end
    #     with adios.file(gridNm) as gridFh:
    #       gridVar = adios.var(gridFh, self._var_name)
    #       offset, count = self._createOffsetCountBp(gridVar, axes, None)
    #       tmp = gridVar.read(offset=offset, count=count)
    #       grid = [tmp[..., d].transpose()
    #               #for d in range(len(cells))]
    #               for d in range(tmp.shape[-1])]
    #       self._grid = grid
    #     #end
    #   elif self._gridType == 'nonuniform':
    #     raise TypeError('\'nonuniform\' is not presently supported')
    #   else:
    #     raise TypeError('Unsupported grid type info in field!')
    #   #end

    # Load data
    num_dims = len(cells)
    var = adios.var(fh, self.var_name)
    grid = [np.linspace(lower[d],
                        upper[d],
                        cells[d]+1)
            for d in range(num_dims)]
    offset, count = self._create_offset_count(var, self.axes, self.comp, grid)
    data = var.read(offset=offset, count=count, nsteps=1)

    # Adjust boundaries for 'offset' and 'count'
    dz = (upper - lower) / cells
    if offset:
      if self.ctx['grid_type']== 'uniform':
        lower = lower + offset[:num_dims]*dz
        cells = cells - offset[:num_dims]
      elif self.ctx['grid_type'] == 'mapped':
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
    return cells, lower, upper, data

  def _read_diagnostic(self) -> tuple:
    import adios
    fh = adios.file(self.file_name)

    time_lst = [key for key, _ in fh.vars.items() if 'TimeMesh' in key]
    data_lst = [key for key, _ in fh.vars.items() if 'Data' in key]
    for i in range(len(data_lst)):
      if i==0:
        data = np.atleast_1d(adios.var(fh, data_lst[i]).read())
        grid = np.atleast_1d(adios.var(fh, time_lst[i]).read())
      else:
        next_data = np.atleast_1d(adios.var(fh, data_lst[i]).read())
        next_grid = np.atleast_1d(adios.var(fh, time_lst[i]).read())
        # deal with weird behavior after restart where some data
        # doesn't have second dimension
        if len(next_data.shape) < 2:
          next_data = np.expand_dims(next_data, axis=1)
        #end
        data = np.append(data, next_data, axis=0)
        grid = np.append(grid, next_grid, axis=0)
      #end
    #end
    fh.close()
    #end

    return [np.squeeze(grid)], [grid[0]], [grid[-1]], data
  #end

  # ---- Exposed function ----------------------------------------------
  def get_data(self,
               grid_file_name : str = None,
               **kwargs) -> tuple:
    grid = None

    if self.is_frame:
      cells, lower, upper, data = self._read_frame()
    #end
    if self.is_diagnostic:
      grid, lower, upper, data = self._read_diagnostic()
      cells = grid[0].shape
    #end

    num_dims = len(cells)
    # Load or construct grid
    if grid_file_name:
      # grid_reader = Read_gkyl(grid_file_name)
      # _, tmp = grid_reader.get_data()
      # num_comps = tmp.shape[-1]
      # num_coeff = num_comps/num_dims
      # grid = [tmp[..., int(d*num_coeff):int((d+1)*num_coeff)]
      #         for d in range(num_dims)]
      if self.ctx:
        self.ctx['grid_type'] = 'c2p'
      #end
    elif grid is None:
      # Create sparse unifrom grid
      # Adjust for ghost cells
      dz = (upper - lower) / cells
      for d in range(num_dims):
        if cells[d] != data.shape[d]:
          ngl = int(np.floor((cells[d] - data.shape[d])*0.5))
          ngu = int(np.ceil((cells[d] - data.shape[d])*0.5))
          cells[d] = data.shape[d]
          lower[d] = lower[d] - ngl*dz[d]
          upper[d] = upper[d] + ngu*dz[d]
        #end
      #end
      grid = [np.linspace(lower[d],
                          upper[d],
                          cells[d]+1)
              for d in range(num_dims)]
      if self.ctx:
        self.ctx['grid_type'] = 'uniform'
      #end
    #end

    return grid, data
  #end