import numpy as np
import os.path
import re

from postgkyl.utils import idxParser

class Read_gkyl_adios(object):
  """Provides a framework to read gkyl Adios output
  """

  def __init__(self,
               file_name : str,
               ctx : dict = None,
               var_name : str = 'CartGridField',
               c2p :str = None,
               axes : tuple = (None, None, None, None, None, None),
               comp : int = None,
               **kwargs) -> None:
    self.file_name = file_name
    self.var_name = var_name
    self.c2p = c2p

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
      import adios2
      fh = adios2.open(self.file_name, "rra")
      for key in fh.available_variables():
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
    #end
    return self.is_frame or self.is_diagnostic
  #end

  def _create_offset_count(self, dims, zs, comp, grid=None) -> tuple:
    num_dims = len(dims)
    count = np.array(dims)
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
    import adios2
    fh = adios2.open(self.file_name, "rra")

    # Postgkyl conventions require the attributes to be
    # narrays even for 1D data
    lower = np.atleast_1d(fh.read_attribute('lowerBounds'))
    upper = np.atleast_1d(fh.read_attribute('upperBounds'))
    cells = np.atleast_1d(fh.read_attribute('numCells'))
    if 'changeset' in fh.available_attributes().keys():
      self.ctx['changeset'] = fh.read_attribute_string('changeset')[0]
    #end
    if 'builddate' in fh.available_attributes().keys():
      self.ctx['builddate'] = fh.read_attribute_string('builddate')[0]
    #end
    if 'polyOrder' in fh.available_attributes().keys():
      self.ctx['polyOrder'] = fh.read_attribute('polyOrder')
      self.ctx['isModal'] = True
    #end
    if 'basisType' in fh.available_attributes().keys():
      self.ctx['basisType'] = fh.read_attribute_string('basisType')[0]
      self.ctx['isModal'] = True
    #end
    if 'charge' in fh.available_attributes().keys():
      self.ctx['charge'] = fh.read_attribute('charge')
    #end
    if 'mass' in fh.available_attributes().keys():
      self.ctx['mass'] = fh.read_attribute('mass')
    #end
    if 'time' in fh.available_variables():
      self.ctx['time'] = fh.read('time')
    #end
    if 'frame' in fh.available_variables():
      self.ctx['frame'] = fh.read('frame')
     #end


    # Load data
    num_dims = len(cells)
    grid = [np.linspace(lower[d],
                        upper[d],
                        cells[d]+1)
            for d in range(num_dims)]
    var_dims = fh.available_variables()[self.var_name]['Shape']
    var_dims = [int(v) for v in var_dims.split(',')]
    offset, count = self._create_offset_count(
            var_dims, self.axes, self.comp, grid)
    data = fh.read(self.var_name, start=offset, count=count)

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

    # Check for mapped grid ...
    #if 'type' in fh.attrs.keys():# and self._comp_grid is False:
    #  self.ctx['grid_type'] = adios.attr(fh, 'type').value.decode('UTF-8')
    #end
    if self.c2p:
      grid_fh = adios2.open(self.c2p, 'r')
      grid_dims = grid_fh.available_variables()['CartGridField']['Shape']
      grid_dims = [int(v) for v in grid_dims.split(',')]
      offset, count = self._create_offset_count(grid_dims, self.axes, None)
      tmp = grid_fh.read('CartGridField', start=offset, count=count)
      num_comps = tmp.shape[-1]
      num_coeff = num_comps/num_dims
      grid = [tmp[..., int(d*num_coeff):int((d+1)*num_coeff)]
              for d in range(num_dims)]
      if self.ctx:
        self.ctx['grid_type'] = 'c2p'
      #end
    # elif 'grid' in fh.attrs.keys():
    #   grid_name = adios.attr(fh, 'grid').value.decode('UTF-8')
    #   grid_fh = adios.file(grid_name)
    #   grid_var = adios.var(grid_fh, 'CartGridField')
    #   offset, count = self._create_offset_count(grid_var, self.axes, None)
    #   tmp = grid_var.read(offset=offset, count=count)
    #   #grid = [tmp[..., d].transpose() for d in range(num_dims)]
    #   grid = [tmp[..., d] for d in range(num_dims)]
    #   if self.ctx:
    #     self.ctx['grid_type'] = 'mapped'
    #   #end
    else:
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

    fh.close()
    return grid, lower, upper, data

  def _read_diagnostic(self) -> tuple:
    import adios2
    fh = adios2.open(self.file_name, "r")

    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    time_lst = [key for key in fh.available_variables() if 'TimeMesh' in key]
    data_lst = [key for key in fh.available_variables() if 'Data' in key]
    time_lst = natural_sort(time_lst)
    data_lst = natural_sort(data_lst)

    for i in range(len(data_lst)):
      if i==0:
        data = np.atleast_1d(fh.read(data_lst[i]))
        grid = np.atleast_1d(fh.read(time_lst[i]))
      else:
        next_data = np.atleast_1d(fh.read(data_lst[i]))
        next_grid = np.atleast_1d(fh.read(time_lst[i]))
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
  def get_data(self) -> tuple:
    grid = None

    if self.is_frame:
      grid, lower, upper, data = self._read_frame()
    #end
    if self.is_diagnostic:
      grid, lower, upper, data = self._read_diagnostic()
      cells = grid[0].shape
    #end

    return grid, data
  #end
