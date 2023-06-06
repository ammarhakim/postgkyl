import click
import numpy as np


def _getGrid(grid0, grid1):
  if grid0 is not None and grid1 is not None:
    if len(grid0) > len(grid1):
      return grid0
    else:
      return grid1
    #end
  elif grid0 is not None:
    return grid0
  elif grid1 is not None:
    return grid1
  else:
    return None
  #end
#end

def add(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  outValues = inValues[0] + inValues[1]
  return [outGrid], [outValues]
#end

def subtract(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  outValues = inValues[1] - inValues[0]
  return [outGrid], [outValues]
#end

def mult(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  a, b = inValues[1], inValues[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    outValues = a*b
  else:
    # When multiplying phase-space and conf-space field, the
    # dimensions do not match. NumPy can do a lot of things with
    # broadcasting
    # (https://numpy.org/doc/stable/user/basics.broadcasting.html) but
    # it requires the trainling indices to match, which is opposite to
    # what we have (the first indices are matching). Therefore, one can
    # transpose, multiply, and transpose back; I think. -- Petr Cagas
    outValues = (a.transpose()*b.transpose()).transpose()
  #end
  return [outGrid], [outValues]
#end

def dot(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  outValues = np.sum(inValues[1]*inValues[0], axis=-1)[..., np.newaxis]
  return [outGrid], [outValues]
#end

def divide(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  a, b = inValues[1], inValues[0]
  if np.array_equal(a.shape, b.shape) or len(a.shape) == 0 or len(b.shape) == 0:
    outValues = a/b
  else:
    # See the 'mult' comment above
    outValues = (a.transpose()/b.transpose()).transpose()
  #end
  return [outGrid], [outValues]
#end

def sqrt(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.sqrt(inValues[0])
  return [outGrid], [outValues]
#end

def psin(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.sin(inValues[0])
  return [outGrid], [outValues]
#end

def pcos(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.cos(inValues[0])
  return [outGrid], [outValues]
#end

def ptan(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.tan(inValues[0])
  return [outGrid], [outValues]
#end

def absolute(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.abs(inValues[0])
  return [outGrid], [outValues]
#end

def log(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.log(inValues[0])
  return [outGrid], [outValues]
#end

def log10(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.log10(inValues[0])
  return [outGrid], [outValues] 
#end

def minimum(inGrid, inValues):
  outValues = np.atleast_1d(np.nanmin(inValues[0]))
  return [[]], [outValues]
#end

def minimum2(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  outValues = np.fmin(inValues[0], inValues[1])
  return [outGrid], [outValues]
#end

def maximum(inGrid, inValues):
  outValues = np.atleast_1d(np.nanmax(inValues[0]))
  return [[]], [outValues] 
#end

def maximum2(inGrid, inValues):
  outGrid = _getGrid(inGrid[0], inGrid[1])
  outValues = np.fmax(inValues[0], inValues[1])
  return [outGrid], [outValues]
#end

def mean(inGrid, inValues):
  outValues = np.atleast_1d(np.mean(inValues[0]))
  return [[]], [outValues]
#end

def power(inGrid, inValues):
  outGrid = inGrid[1]
  outValues = np.power(inValues[1], inValues[0])
  return [outGrid], [outValues] 
#end

def sq(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = inValues[0]**2
  return [outGrid], [outValues]
#end

def exp(inGrid, inValues):
  outGrid = inGrid[0]
  outValues = np.exp(inValues[0])
  return [outGrid], [outValues]
#end

def length(inGrid, inValues):
  ax = int(inValues[0])
  length = inGrid[1][ax][-1] - inGrid[1][ax][0]
  if len(inGrid[1][ax]) == inValues[1].shape[ax]:
    length = length + inGrid[1][ax][1] - inGrid[1][ax][0]
  #end  
  return [[]], [length]

def grad(inGrid, inValues):
  out_grid = inGrid[0]
  nd = len(inValues[0].shape)-1
  out_shape = list(inValues[0].shape)
  nc = inValues[0].shape[-1]
  out_shape[-1] = nc*nd
  out_values = np.zeros(out_shape)
  
  for d in range(nd):
    zc = 0.5*(inGrid[0][d][1:] + inGrid[0][d][:-1]) # get cell centered values
    out_values[..., d*nc:(d+1)*nc] = np.gradient(inValues[0],
                                                 zc,
                                                 edge_order=2,
                                                 axis=d)
  #end
  return [out_grid], [out_values]
#end

def grad2(inGrid, inValues):
  outGrid = inGrid[1]
  ax = inValues[0]
  if isinstance(ax, str) and ':' in ax:
    tmp = ax.split(':')
    lo = int(tmp[0])
    up = int(tmp[1])
    rng = range(lo, up)
  elif isinstance(ax, str):
    rng = tuple((int(i) for i in ax.split(',')))
  else:
    rng = range(int(ax), int(ax+1))
  #end
    
  numDims = len(rng)
  outShape = list(inValues[1].shape)
  numComps = inValues[1].shape[-1]
  outShape[-1] = outShape[-1]*numDims
  outValues = np.zeros(outShape)
  
  for cnt, d in enumerate(rng):
    zc = 0.5*(inGrid[1][d][1:] + inGrid[1][d][:-1]) # get cell centered values
    outValues[..., cnt*numComps:(cnt+1)*numComps] = np.gradient(inValues[1],
                                                                zc,
                                                                edge_order=2,
                                                                axis=d)
  #end
  return [outGrid], [outValues]
#end

def integrate(inGrid, inValues, avg=False):
  grid = inGrid[1].copy()
  values = np.array(inValues[1])
  
  axis = inValues[0]
  if isinstance(axis, float):
    axis = tuple([int(axis)])
  elif isinstance(axis, tuple):
    pass
  elif isinstance(axis, np.ndarray):
    axis = tuple([int(axis)])
  elif isinstance(axis, str):
    if len(axis.split(',')) > 1:
      axes = axis.split(',')
      axis = tuple([int(a) for a in axes])
    elif len(axis.split(':')) == 2:
      bounds = axis.split(':')
      axis = tuple(range(bounds[0], bounds[1]))
    elif axis == 'all':
      numDims = len(grid)
      axis = tuple(range(numDims))
    #end
  else:
    raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")
  #end
    
  dz = []
  for d, coord in enumerate(grid):
    dz.append(coord[1:] - coord[:-1])
    if len(coord) == values.shape[d]:
      dz[-1] = np.append(dz[-1], dz[-1][-1])
    #end

  # Integration assuming values are cell centered averages
  # Should work for nonuniform meshes
  for ax in sorted(axis, reverse=True):
    values = np.moveaxis(values, ax, -1)
    values = np.dot(values, dz[ax])
  #end
  for ax in sorted(axis):
    grid[ax] = np.array([0])
    values = np.expand_dims(values, ax)
    if avg:
      length = inGrid[1][ax][-1] - inGrid[1][ax][0]
      if len(inGrid[1][ax]) == inValues[1].shape[ax]:
        length = length + inGrid[1][ax][1] - inGrid[1][ax][0]
      #end
      values = values / length
    #end
  #end
  return [grid], [values]
#end

def average(inGrid, inValues):
  return integrate(inGrid, inValues, True)

def divergence(inGrid, inValues):
  outGrid = inGrid[0]
  numDims = len(inGrid[0])
  numComps = inValues[0].shape[-1]
  if numComps > numDims:
    click.echo(click.style("WARNING in 'ev div': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). The last {:d} component(s) of the vector will be disregarded.".format(numComps, numDims, numComps-numDims), fg='yellow'))
    #end
  outShape = list(inValues[0].shape)
  outShape[-1] = 1
  outValues = np.zeros(outShape)
  for d in range(numDims):
    zc = 0.5*(inGrid[0][d][1:] + inGrid[0][d][:-1]) # get cell centered values
    outValues[..., 0] = outValues[..., 0] + np.gradient(inValues[0][..., d], zc, edge_order=2, axis=d)
  #end
  return [outGrid], [outValues]
#end

def curl(inGrid, inValues):
  outGrid = inGrid[0]
  numDims = len(inGrid[0])
  numComps = inValues[0].shape[-1]
  
  outShape = list(inValues[0].shape)

  if numDims == 1:
    if numComps != 3:
      raise ValueError("ERROR in 'ev curl': Curl in 1D requires 3-component input and {:d}-component field was provided.".format(numComps))
    #end
    zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
    outValues = np.zeros(outShape)
    outValues[..., 1] = - np.gradient(inValues[0][..., 2], zc0, edge_order=2, axis=0)
    outValues[..., 2] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0)
  elif numDims == 2:
    zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
    zc1 = 0.5*(inGrid[0][1][1:] + inGrid[0][1][:-1])
    if numComps < 2:
      raise ValueError("ERROR in 'ev curl': Length of the provided vector ({:d}) is smaller than number of dimensions ({:d}). Curl can't be calculated.".format(numComps, numDims))
    elif numComps == 2:
      click.echo(click.style("WARNING in 'ev curl': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). Only the third component of curl will be calculated.".format(numComps, numDims), fg='yellow'))
      outShape[-1] = 1
      outValues = np.zeros(outShape)
      outValues[..., 0] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)
    else:
      if numComps > 3:
        print("here")
        click.echo(click.style("WARNING in 'ev curl': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). The last {:d} components of the vector will be disregarded.".format(numComps, numDims, numComps-numDims), fg='yellow'))
      #end
      outValues = np.zeros(outShape)
      outValues[..., 0] = np.gradient(inValues[0][..., 2], zc1, edge_order=2, axis=1)
      outValues[..., 1] = - np.gradient(inValues[0][..., 2], zc0, edge_order=2, axis=0)
      outValues[..., 2] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)
  else: # 3D
    if numComps > 3:
      click.echo(click.style("WARNING in 'ev curl': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). The last {:d} component(s) of the vector will be disregarded.".format(numComps, numDims, numComps-numDims), fg='yellow'))
    elif numComps < 3:
      raise ValueError("ERROR in 'ev curl': Length of the provided vector ({:d}) is smaller than number of dimensions ({:d}). Curl can't be calculated".format(numComps, numDims))
    #end
    zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
    zc1 = 0.5*(inGrid[0][1][1:] + inGrid[0][1][:-1])
    zc2 = 0.5*(inGrid[0][2][1:] + inGrid[0][2][:-1])
    outValues[..., 0] = np.gradient(inValues[0][..., 2], zc1, edge_order=2, axis=1) - np.gradient(inValues[0][..., 1], zc2, edge_order=2, axis=2)
    outValues[..., 1] = np.gradient(inValues[0][..., 0], zc2, edge_order=2, axis=2) - np.gradient(inValues[0][..., 2], zc0, edge_order=2, axis=0)
    outValues[..., 2] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)
  #end
  return [outGrid], [outValues]
#end

cmds = { '+' : { 'numIn' : 2, 'numOut' : 1, 'func' : add }, 
         '-' : { 'numIn' : 2, 'numOut' : 1, 'func' : subtract },
         '*' : { 'numIn' : 2, 'numOut' : 1, 'func' : mult },
         '/' : { 'numIn' : 2, 'numOut' : 1, 'func' : divide },
         'dot' : { 'numIn' : 2, 'numOut' : 1, 'func' : dot },
         'sqrt' : { 'numIn' : 1, 'numOut' : 1, 'func' : sqrt },
         'sin' : { 'numIn' : 1, 'numOut' : 1, 'func' : psin },
         'cos' : { 'numIn' : 1, 'numOut' : 1, 'func' : pcos },
         'tan' : { 'numIn' : 1, 'numOut' : 1, 'func' : ptan },
         'abs' : { 'numIn' : 1, 'numOut' : 1, 'func' : absolute },
         'avg' : { 'numIn' : 2, 'numOut' : 1, 'func' : average },
         'log' : { 'numIn' : 1, 'numOut' : 1, 'func' : log },
         'log10' : { 'numIn' : 1, 'numOut' : 1, 'func' : log10 },
         'max' : { 'numIn' : 1, 'numOut' : 1, 'func' : maximum },
         'min' : { 'numIn' : 1, 'numOut' : 1, 'func' : minimum },
         'max2' : { 'numIn' : 2, 'numOut' : 1, 'func' : maximum2 },
         'min2' : { 'numIn' : 2, 'numOut' : 1, 'func' : minimum2 },
         'mean' : { 'numIn' : 1, 'numOut' : 1, 'func' : mean },
         'len' : { 'numIn' : 2, 'numOut' : 1, 'func' : length },
         'pow' : { 'numIn' : 2, 'numOut' : 1, 'func' : power },
         'sq' : { 'numIn' : 1, 'numOut' : 1, 'func' : sq },
         'exp' : { 'numIn' : 1, 'numOut' : 1, 'func' : exp },
         'grad' : { 'numIn' : 1, 'numOut' : 1, 'func' : grad },
         'grad2' : { 'numIn' : 2, 'numOut' : 1, 'func' : grad2 },
         'int' : { 'numIn' : 2, 'numOut' : 1, 'func' : integrate },
         'div' : { 'numIn' : 1, 'numOut' : 1, 'func' : divergence },
         'curl' : { 'numIn' : 1, 'numOut' : 1, 'func' : curl },
}
