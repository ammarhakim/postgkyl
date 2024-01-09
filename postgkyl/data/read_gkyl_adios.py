import numpy as np
import click
import re

from postgkyl.utils import idxParser

class Read_gkyl_adios(object):
  """Provides a framework to read gkyl Adios output
  """

  def __init__(self,
               file_name: str,
               ctx: dict = None,
               var_name: str = 'CartGridField',
               c2p: str = None,
               axes: tuple = (None, None, None, None, None, None),
               comp: int = None,
               click_mode: bool = False,
               **kwargs) -> None:
    self._file_name = file_name
    self.var_name = var_name
    self.c2p = c2p

    self.axes = axes
    self.comp = comp

    self.lower = None
    self.upper = None
    self.num_comps = None
    self.cells = None

    self.is_frame = False
    self.is_diagnostic = False
    self.click_mode = click_mode

    self.ctx = ctx
  #end

  def _is_compatible(self) -> bool:
    try:
      import adios2 # Adios has been a problematic dependency;
        # therefore it is only imported when actially needed
      fh = adios2.open(self._file_name, 'rra')
      for vn in fh.available_variables():
        if 'TimeMesh' in vn:
          self.is_diagnostic = True
          fh.close()
          return True
        #end
      #end

      available_var_names = ''
      for vn in fh.available_variables():
        available_var_names += '\'{:s}\', '.format(str(vn))
      #end
      if self.var_name not in fh.available_variables():
        self.ctx['var_names'] = available_var_names[:-2]
      #end
      self.is_frame = True
      fh.close()
      return True
    except:
      return False
    #end
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

  def _preload_frame(self) -> None:
    import adios2
    fh = adios2.open(self._file_name, 'rra')

    # Postgkyl conventions require the attributes to be
    # narrays even for 1D data
    self.lower = np.atleast_1d(fh.read_attribute('lowerBounds'))
    self.upper = np.atleast_1d(fh.read_attribute('upperBounds'))
    self.cells = np.atleast_1d(fh.read_attribute('numCells'))
    if 'changeset' in fh.available_attributes().keys():
      self.ctx['changeset'] = fh.read_attribute_string('changeset')[0]
    #end
    if 'builddate' in fh.available_attributes().keys():
      self.ctx['builddate'] = fh.read_attribute_string('builddate')[0]
    #end
    if 'polyOrder' in fh.available_attributes().keys():
      self.ctx['poly_order'] = fh.read_attribute('polyOrder')[0]
      self.ctx['is_modal'] = True
    #end
    if 'basisType' in fh.available_attributes().keys():
      self.ctx['basis_type'] = fh.read_attribute_string('basisType')[0]
      self.ctx['is_modal'] = True
    #end
    if 'charge' in fh.available_attributes().keys():
      self.ctx['charge'] = fh.read_attribute('charge')[0]
    #end
    if 'mass' in fh.available_attributes().keys():
      self.ctx['mass'] = fh.read_attribute('mass')[0]
    #end
    if 'time' in fh.available_variables():
      self.ctx['time'] = fh.read('time')
    #end
    if 'frame' in fh.available_variables():
      self.ctx['frame'] = fh.read('frame')
    #end

    fh.close()
  #end

  def _load_frame(self) -> tuple:
    import adios2
    fh = adios2.open(self._file_name, 'rra')

    if self.var_name not in fh.available_variables():
      if self.click_mode:
        var_name = self.var_name
        while True:
          var_name = click.prompt('Variable name \'{:s}\' is not available please select from the available ones: {:s}'.format(var_name, self.ctx['var_names']))
          if var_name in fh.available_variables():
            self.var_name = var_name
            self.ctx.pop('var_names', None)
            break
          #end
        #end
      else:
        raise ValueError('Could not find the variable \'{:s}\'; available variables are: {:s}'. format(var_name, self.ctx['var_names']))
      #end
    #end

    num_dims = len(self.cells)
    grid = [np.linspace(self.lower[d],
                        self.upper[d],
                        self.cells[d]+1)
            for d in range(num_dims)]
    var_dims = fh.available_variables()[self.var_name]['Shape']
    var_dims = [int(v) for v in var_dims.split(',')]
    offset, count = self._create_offset_count(
      var_dims, self.axes, self.comp, grid)
    data = fh.read(self.var_name, start=offset, count=count)

    # Adjust boundaries for 'offset' and 'count'
    dz = (self.upper - self.lower) / self.cells
    if offset:
      if self.ctx['grid_type'] == 'uniform':
        self.lower = self.lower + offset[:num_dims]*dz
        self.cells = self.cells - offset[:num_dims]
      elif self.ctx['grid_type'] == 'mapped':
        idx = np.full(num_dims, 0)
        for d in range(num_dims):
          self.lower[d] = self._grid[d][tuple(idx)]
          self.cells[d] = self.cells[d] - offset[d]
        #end
      #end
    #end
    if count:
      if self.ctx['grid_type']  == 'uniform':
        self.upper = self.lower + count[:num_dims]*dz
        self.cells = count[:num_dims]
      elif self.ctx['grid_type']  == 'mapped':
        idx = np.full(num_dims, 0)
        for d in range(num_dims):
          idx[-d-1] = count[d]-1  #.Reverse indexing of idx because of transpose() in composing self._grid.
          self.upper[d] = self._grid[d][tuple(idx)]
          self.cells[d] = count[d]
        #end
      #end
    #end

    # Check for mapped grid ...
    #if 'type' in fh.attrs.keys():# and self._comp_grid is False:
    #  self.ctx['grid_type'] = adios.attr(fh, 'type').value.decode('UTF-8')
    #end
    if self.c2p:
      grid_fh = adios2.open(self.c2p, 'rra')
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

    fh.close()
    return grid, data

  def _load_diagnostic(self) -> tuple:
    import adios2
    fh = adios2.open(self._file_name, 'rra')

    def natural_sort(l):
      convert = lambda text: int(text) if text.isdigit() else text.lower()
      alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
      return sorted(l, key=alphanum_key)
    #end

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

    return [np.squeeze(grid)], data
  #end

  # ---- Exposed function ----------------------------------------------
  def preload(self) -> None:
    if self.is_frame:
      self._preload_frame()
      if self.ctx:
        self.ctx['cells'] = self.cells
        self.ctx['lower'] = self.lower
        self.ctx['upper'] = self.upper
      #end
    #end
  #end

  def load(self) -> tuple:
    grid, data = None, None

    if self.is_frame:
      grid, data = self._load_frame()
    #end
    if self.is_diagnostic:
      grid, data = self._load_diagnostic()
    #end

    self.ctx['num_comps'] = data.shape[-1]

    return grid, data
  #end
