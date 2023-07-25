import os.path
from glob import glob

import tables
import numpy as np

from postgkyl.data import GData
from postgkyl.data.computeInterpolationMatrices import createInterpMatrix
from postgkyl.data.computeDerivativeMatrices import createDerivativeMatrix

from postgkyl.data.recovData import recovC0Fn, recovC1Fn, recovEdFn

path = os.path.dirname(os.path.realpath(__file__))

numNodesSerendipity = np.array([[1,  2,   3,   4,   5],
                                [1,  4,   8,  12,  17],
                                [1,  8,  20,  32,  50],
                                [1, 16,  48,  80, 136],
                                [1, 32, 112, 192, 352],
                                [1, 64, 256, 448, 880]])
numNodesMaximal = np.array([[2,  3,  4,   5],
                            [3,  6, 10,  15],
                            [4, 10, 20,  35],
                            [5, 15, 35,  70],
                            [6, 21, 56, 126],
                            [7, 28, 84, 210]])
numNodesTensor = np.array([[ 2,   3,    4,     5],
                           [ 4,   9,   16,    25],
                           [ 8,  27,   64,   125],
                           [16,  81,  256,   625],
                           [32, 343, 1024,  3125],
                           [64, 729, 4096, 15625]])
numNodesGkHybrid = np.array([1, 6, 12, 24, 48])
numNodesPkpmHybrid = np.array([1, 6, 12, 24, 48])

def _get_basis_p(num_dim, num_comp):
  basis, poly_order = None, None
  idx = np.argwhere(numNodesSerendipity[num_dim-1, :] == num_comp).squeeze()
  if idx:
    basis = 'serendipity'
    poly_order = idx
  #end
  idx = np.argwhere(numNodesTensor[num_dim-1, :] == num_comp).squeeze()
  if idx:
    basis = 'tensor'
    poly_order = idx+1
  #end
  return basis, poly_order
#end

def _getNumNodes(dim, polyOrder, basisType):
  if basisType.lower() == 'serendipity':
    numNodes = numNodesSerendipity[dim-1, polyOrder]
  elif basisType.lower() == 'maximal-order':
    numNodes = numNodesMaximal[dim-1, polyOrder-1]
  elif basisType.lower() == 'tensor':
    numNodes = numNodesTensor[dim-1, polyOrder-1]
  elif basisType.lower() == 'gkhybrid':
    numNodes = numNodesGkHybrid[dim-1]
  elif basisType.lower() == 'pkpmhybrid':
    numNodes = numNodesPkpmHybrid[dim-1]
  else:
    raise NameError(
      "GInterp: Basis '{:s}' is not supported!\n"
      "Supported basis are currently 'ns' (Nodal Serendipity),"
      " 'ms' (Modal Serendipity), 'mt' (Modal Tensor product),"
      " 'mo' (Modal maximal Order), 'gkhybrid' (Modal GkHybrid),"
      " and 'pkpmhybrid' (Modal PKPM hybrid)".
      format(basisType))
  #end
  return numNodes
#end


def _loadInterpMatrix(dim, polyOrder, basisType, interp, read, modal, c2p=False):
  if (interp is not None and read is None) or c2p:
    if interp is None:
      interp = polyOrder+1
    #end
    mat = createInterpMatrix(dim, polyOrder, basisType, interp, modal, c2p)
    return mat
  elif basisType=='tensor':
    mat = createInterpMatrix(dim, polyOrder, 'tensor', polyOrder+1, True, c2p)
    return mat
  elif basisType=='gkhybrid':
    mat = createInterpMatrix(dim, polyOrder, 'gkhybrid', polyOrder+1, True, c2p)
    return mat
  elif basisType=='pkpmhybrid':
    mat = createInterpMatrix(dim, polyOrder, 'pkpmhybrid', polyOrder+1, True, c2p)
    return mat
  else:
    # Load interpolation matrix from the pre-computed HDF5 file.
    varid = 'xformMatrix%i%i' % (dim, polyOrder)
    if modal == False and basisType.lower() == 'serendipity':
      fileName = path + '/xformMatricesNodalSerendipity.h5'
    elif modal and basisType.lower() == 'serendipity':
      fileName = path + '/xformMatricesModalSerendipity.h5'

    elif modal and basisType.lower() == 'maximal-order':
      fileName = path + '/xformMatricesModalMaximal.h5'
    else:
      raise NameError(
        "GInterp: Basis {:s} is not supported!\n"
        "Supported basis are currently 'ns' (Nodal Serendipity), "
        "'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
        format(basisType))
    #end
    fh = tables.open_file(fileName)
    mat = fh.root.matrices._v_children[varid].read()
    fh.close()
    return mat.transpose()
  #end
#end


def _loadDerivativeMatrix(dim, polyOrder, basisType, interp, read, modal=True):
  if interp is not None and read is None:
    mat = createDerivativeMatrix(dim, polyOrder, basisType, interp, modal)
    return mat
  else:
    interp = polyOrder+1
    mat = createDerivativeMatrix(dim, polyOrder, basisType, interp, modal)
    return mat
  #end
#end


def _makeMesh(nInterp, Xc, xlo=None, xup=None, gridType=None):
  nx = Xc.shape[0]-1 # expecting nodal mesh
  meshOut = np.zeros(nInterp*nx+1)
  if gridType is None or gridType=="uniform":
    if xlo is None or xup is None:
      xlo = Xc[0]
      xup = Xc[-1]
    #end
    meshOut = np.linspace(xlo, xup, nInterp*nx+1)
  elif gridType=="mapped":
    # subdivide every cell in Xc into nInterp cells.
    for i in range(nx):
      dx = (Xc[i+1]-Xc[i])/nInterp
      for j in range(nInterp):
        meshOut[i*nInterp+j] = Xc[i]+j*dx
      #end
    #end
    # add the last node.
    dx = (Xc[-1]-Xc[-2])/nInterp
    meshOut[nx*nInterp] = Xc[nx-1]+nInterp*dx
  #end
  return meshOut
#end

def _make1Dgrids(nInterp, Xc, numDims, gridType=None):
  # build a list of 1D arrays, each containing the grid in that dimension.
  gridOut = list()
  if gridType is None or gridType=="uniform":
    gridOut = [_makeMesh(nInterp[d], Xc[d])
               for d in range(numDims)]
  elif gridType=="mapped":
    # back out 1D arrays from Xc.
    for d in range(numDims):
      currSlices = [0]*numDims
      currSlices[-1-d] = np.s_[:]
      gridOut.append(_makeMesh(nInterp[d], Xc[d][tuple(currSlices)],gridType=gridType))
    #end
  #end
  return gridOut
#end

def _interpOnMesh(cMat, qIn, nInterpIn, basisType, c2p=False):
  shift = 0
  print(cMat.shape, qIn.shape)
  numCells = np.array(qIn.shape)
  # last entry is indexing nodes, get rid of it
  numCells = numCells[:-1]
  numDims = int(len(numCells))
  numInterp = np.array([max(nInterpIn, 2)]*numDims)
  if basisType == "gkhybrid":
    # 1x1v, 1x2v, 2x2v, 3x2v cases, with p=2 in the first velocity dim.
    vpardir = 1 if (numDims==2 or numDims==3) else (2 if numDims==4  else (3 if numDims==5 else 99))
    numInterp[vpardir] = nInterpIn+1
  #end
  if basisType == "pkpmhybrid":
    numInterp[-1] = nInterpIn+1
  #end
  if c2p:
    qOut = np.zeros(numCells*(numInterp)+1, np.float64)
  else:
    qOut = np.zeros(numCells*numInterp, np.float64)
  #end
  # move the node index from last to the first
  qIn = np.moveaxis(qIn, -1, 0)
  # Main loop
  for n in range(np.prod(numInterp)):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
    temp = np.tensordot(cMat[n, :], qIn, axes=1)
    # decompose n to i,j,k,... indices based on the number of dimensions
    startIdx = np.unravel_index(n, numInterp, order='F')
    # define multi-D qOut slices
    if c2p:
      idxs = [slice(int(startIdx[i]), int(numCells[i]*(numInterp[i]-1)+startIdx[i]), numInterp[i]-1)
              for i in range(numDims)]
    else:
      idxs = [slice(int(startIdx[i]), int(numCells[i]*numInterp[i]), numInterp[i])
              for i in range(numDims)]
    #end
    qOut[tuple(idxs)] = temp
  #end
  return np.array(qOut)
#end



class GInterp(object):
  """Postgkyl base class for DG data manipulation.

  This class should not be used on its own! Currently supported
  child classes are:
    - GInterpNodal
    - GInterpModal

  Init Args:
    data (GData): Data to work with
    numNodes (int): Number of nodes
  """

  def __init__(self, data, numNodes):
    self.data = data
    self.numNodes = numNodes
    self.numEqns = data.getNumComps()/numNodes
    self.numDims = data.getNumDims()
    self.Xc = data.getGrid()
    self.gridType = data.getGridType()
  #end

  def _getRawNodal(self, component):
    q = self.data.getValues()
    numEqns = self.numEqns
    shp = [q.shape[i] for i in range(self.numDims)]
    shp.append(self.numNodes)
    rawData = np.zeros(shp, np.float64)
    for n in range(self.numNodes):
      rawData[..., n] = q[..., int(component+n*numEqns)]
    #end
    return rawData
  #end

  def _getRawModal(self, component):
    q = self.data.getValues()
    numEqns = self.numEqns
    shp = [q.shape[i] for i in range(self.numDims)]
    shp.append(self.numNodes)
    rawData = np.zeros(shp, np.float64)
    lo = int(component*self.numNodes)
    up = int(lo+self.numNodes)
    rawData = q[..., lo:up]
    return rawData
  #end
#end

class GInterpNodal(GInterp):
  """Postgkyl class for nodal DG data manipulation.

  After the initializations, GInterpNodal object provides the
  interpolate and differentiate methods.  These returns grid and
  values by default but could be used to directly push to the GData
  stack with the stack=True flag.

  Parent: GInterp

  Init Args:
    data (GData): Data to work with
    polyOrder (int): Order of the polynomial approximation
    basis (str): Specify the basis. Currently supported is the
      nodal Serendipity 'ns'
    numInterp (int): Specify number of points on which to
      interpolate (default: polyOrder + 1)
    read

  Example:
    import postgkyl
    data = postgkyl.GData('file.h5')
    dg = postgkyl.GInterpNodal(data, 2, 'ns')
    grid, values = dg.interpolate()
  """

  def __init__(self, data, polyOrder, basisType,
               numInterp=None, read=None):
    self.numDims = data.getNumDims()
    self.polyOrder = polyOrder
    self.basisType = basisType
    if basisType == 'ns':
      self.basisType = 'serendipity'
    #end

    self.numInterp = numInterp
    self.read = read
    numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basisType)
    GInterp.__init__(self, data, numNodes)
  #end

  def interpolate(self, comp=0, overwrite=False, stack=False):
    if stack:
      overwrite = stack
      print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    cMat = _loadInterpMatrix(self.numDims, self.polyOrder,
                             self.basisType, self.numInterp, self.read, False)
    if isinstance(comp, int):
      q = self._getRawNodal(comp)
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
    elif isinstance(comp, tuple):
      q = self._getRawNodal(comp[0])
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
      for c in comp[1:]:
        q = self._getRawNodal(c)
        values = np.append(values,
                           _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis],
                           axis=-1)
      #end
    elif isinstance(comp, slice):
      q = self._getRawNodal(comp.start)
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
      for c in range(comp.start+1, comp.stop):
        q = self._getRawNodal(c)
        values = np.append(values,
                           _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis],
                           axis=-1)
      #end
    #end

    nInterp = [int(round(cMat.shape[0] ** (1.0/self.numDims)))]*self.numDims
    grid = _make1Dgrids(nInterp, self.Xc, self.numDims)
    if overwrite:
      self.data.push(grid, values)
    else:
      return grid, values
    #end
  #end

  def differentiate(self, direction, comp=0, overwrite=False, stack=False):
    if stack:
      overwrite = stack
      print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    q = self._getRawNodal(comp)
    cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder,
                                 self.basisType, self.numInterp, self.read, False)
    if direction is not None:
      values = _interpOnMesh(cMat[:, :, direction], q, self.numInterp, self.basisType) \
              * 2/(self.Xc[direction][1]-self.Xc[direction][0])
      values = values[..., np.newaxis]
    else:
      values = np.zeros(q.shape, self.numDims)
      for i in range(self.numDims):
        values[:,i] = _interpOnMesh(cMat[:,:,i], q, self.numInterp, self.basisType)
        values[:,i] *= 2/(self.Xc[i][1]-self.Xc[i][0])
      #end
    #end

    nInterp = [int(round(cMat.shape[0] ** (1.0/self.numDims)))]*self.numDims
    grid = _make1Dgrids(nInterp, self.Xc, self.numDims)
    if overwrite:
      self.data.push(grid, values)
    else:
      return grid, values
    #end
  #end
#end


class GInterpModal(GInterp):
  """Postgkyl class for modal DG data manipulation.

  After the initializations, GInterpModal object provides the
  interpolate and differentiate methods.  These returns grid and
  values by default but could be used to directly push to the GData
  stack with the stack=True flag.

  Parent: GInterp

  Init Args:
    data (GData): Data to work with
    polyOrder (int): Order of the polynomial approximation
    basis (str): Specify the basis. Currently supported are the
      modal Serendipity 'ms' and the maximal order basis 'mo'
    numInterp (int): Specify number of points on which to
      interpolate (default: polyOrder + 1)
    read

  Example:
    import postgkyl
    data = postgkyl.GData('file.bp')
    dg = postgkyl.GInterpModal(data, 2, 'ms')
    grid, values = dg.interpolate()
  """

  def __init__(self, data, polyOrder=None, basisType=None,
               numInterp=None, periodic=False, read=None):
    self.numDims = data.getNumDims()
    if polyOrder is not None:
      self.polyOrder = polyOrder
    elif data.meta['polyOrder'] is not None:
      self.polyOrder = data.meta['polyOrder']
    else:
      raise ValueError(
        'GInterpNodal: polynomial order is neither specified nor stored in the output file')
    #end
    if basisType is not None:
      if basisType == 'ms':
        self.basisType = 'serendipity'
      elif basisType == 'mo':
        self.basisType = 'maximal-order'
      elif basisType == 'mt':
        self.basisType = 'tensor'
      elif basisType == 'gkhyb':
        self.basisType = 'gkhybrid'
      elif basisType == 'pkpmhyb':
        self.basisType = 'pkpmhybrid'
      #end
    elif data.meta['basisType'] is not None:
      self.basisType = data.meta['basisType']
    else:
      raise ValueError('GInterpModal: basis type is neither specified nor stored in the output file')
    #end

    # PKPM hybrid base expects 2+ dimensions with the last one being
    # the parallel velocity. This allows to specify 'pkpmhyb' basis
    # and work with 1x1v and 1x data simulataneously.
    if self.numDims == 1 and self.basisType == 'pkpmhybrid':
      self.basisType = 'serendipity'
    #end

    self.periodic = periodic
    if numInterp is not None and self.polyOrder > 1:
      self.numInterp = numInterp
    else:
      self.numInterp = self.polyOrder + 1
    #end
    self.read = read
    numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basisType)
    GInterp.__init__(self, data, numNodes)
  #end

  def interpolate(self, comp=0, overwrite=False, stack=False):
    if stack:
      overwrite = stack
      print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    cMat = _loadInterpMatrix(self.numDims, self.polyOrder,
                             self.basisType, self.numInterp, self.read, True)
    if isinstance(comp, int):
      q = self._getRawModal(comp)
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
    elif isinstance(comp, tuple):
      q = self._getRawModal(comp[0])
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
      for c in comp[1:]:
        q = self._getRawModal(c)
        values = np.append(values,
                           _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis],
                           axis=-1)
      #end
    elif isinstance(comp, slice):
      q = self._getRawModal(comp.start)
      values = _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis]
      for c in range(comp.start+1, comp.stop):
        q = self._getRawModal(c)
        values = np.append(values,
                           _interpOnMesh(cMat, q, self.numInterp, self.basisType)[..., np.newaxis],
                           axis=-1)
      #end
    #end

    if self.data.meta['grid_type'] == 'c2p':
      q = self.data.getGrid()
      num_comp = q[0].shape[-1]
      basis, poly_order = _get_basis_p(self.numDims, num_comp)
      cMat = _loadInterpMatrix(self.numDims, poly_order,
                               basis, self.numInterp, self.read, True, True)
      grid = []
      for d in range(self.numDims):
        grid.append(_interpOnMesh(cMat, q[d], self.numInterp, basis, True))
      #end
    else:
      if self.basisType == "gkhybrid":
        # 1x1v, 1x2v, 2x2v, 3x2v cases, with p=2 in the first velocity dim.
        vpardir = 1 if (self.numDims==2 or self.numDims==3) else (2 if self.numDims==4  else (3 if self.numDims==5 else 99))
        nInterp = [self.numInterp]*self.numDims
        nInterp[vpardir] = self.numInterp+1
      elif self.basisType == 'pkpmhybrid':
        nInterp = [self.numInterp]*self.numDims
        nInterp[-1] = self.numInterp+1
      else:
        nInterp = [int(round(cMat.shape[0] ** (1.0/self.numDims)))]*self.numDims
      grid = _make1Dgrids(nInterp, self.Xc, self.numDims, self.gridType)
    #end
    if overwrite:
      self.data.push(grid, values)
    else:
      return grid, values
    #end
  #end

  def interpolateGrid(self, overwrite=False):
    if self.data.meta['grid_type'] == 'c2p':
      q = self.data.getGrid()
      num_comp = q[0].shape[-1]
      basis, poly_order = _get_basis_p(self.numDims, num_comp)
      cMat = _loadInterpMatrix(self.numDims, poly_order,
                               basis, self.numInterp, self.read, True, True)
      grid = []
      for d in range(self.numDims):
        grid.append(_interpOnMesh(cMat, q[d], self.numInterp, self.basisType, True))
      #end
    else:
      nInterp = [self.numInterp]*self.numDims
      grid = _make1Dgrids(nInterp, self.Xc, self.numDims, self.gridType)
    #end

    if overwrite:
      self.data.setGrid(grid)
    else:
      return grid
    #end
  #end

  def differentiate(self, direction=None, comp=0, overwrite=False, stack=False):
    if stack:
      overwrite = stack
      print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    q = self._getRawModal(comp)
    cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder,
                                 self.basisType, self.numInterp, self.read, True)
    if direction is not None:
      values = _interpOnMesh(cMat[:, :, direction], q, self.numInterp, self.basisType) \
              * 2/(self.Xc[direction][1]-self.Xc[direction][0])
      values = values[..., np.newaxis]
    else:
      values = _interpOnMesh(cMat[...,0], q, self.numInterp, self.basisType)
      values /= (self.Xc[0][1]-self.Xc[0][0])
      values = values[..., np.newaxis]
      for i in range(1, self.numDims):
        values = np.append(values, _interpOnMesh(cMat[...,i], q, self.numInterp, self.basisType)[...,np.newaxis], axis=self.numDims)
        values[...,i] *= 2/(self.Xc[i][1]-self.Xc[i][0])
      #end
    #end

    nInterp = [int(round(cMat.shape[0] ** (1.0/self.numDims)))]*self.numDims
    grid = _make1Dgrids(nInterp, self.Xc, self.numDims, self.gridType)
    if overwrite:
      self.data.push(grid, values)
    else:
      return grid, values
    #end
  #end

  def recovery(self, comp=0, c1=False, overwrite=False, stack=False):
    if stack:
      overwrite = stack
      print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    if isinstance(comp, int):
      q = self._getRawModal(comp)
    else:
      raise ValueError("recovery: only 'int' comp implemented so far")
    #end
    if self.numDims > 1:
      raise ValueError("recovery: only 1D implemented so far")
    #end

    if self.numInterp is not None:
      N = self.numInterp
    else:
      N = 100
    #end

    numCells = self.data.getNumCells()
    grid = [np.linspace(self.Xc[int(d)][0], self.Xc[int(d)][-1], int(numCells*N+1))
            for d in range(self.numDims)]

    values = np.zeros(numCells*N)
    dx = (self.Xc[0][1]-self.Xc[0][0])

    xC = np.linspace(-1, 1, N, endpoint=False)*dx/2
    xL = np.linspace(-1, 0, N, endpoint=False)*dx
    xR = np.linspace(0, 1, N, endpoint=False)*dx

    if self.periodic:
      if c1:
        values[:N] = recovC1Fn[self.polyOrder-1](xC, q[0],q[-1],q[1], dx)
        values[-N:] = recovC1Fn[self.polyOrder-1](xC, q[-1],q[-2],q[0], dx)
      else:
        values[:N] = recovC0Fn[self.polyOrder-1](xC, q[0],q[-1],q[1], dx)
        values[-N:] = recovC0Fn[self.polyOrder-1](xC, q[-1],q[-2],q[0], dx)
      #end
    else:
      values[:N] = recovEdFn[self.polyOrder-1](xL, q[0], q[1], dx)
      values[-N:] = recovEdFn[self.polyOrder-1](xR, q[-2], q[-1], dx)
    #end
    for j in range(1, numCells[0]-1):
      if c1:
        values[j*N:(j+1)*N] = recovC1Fn[self.polyOrder-1](xC, q[j], q[j-1], q[j+1], dx)
      else:
        values[j*N:(j+1)*N] = recovC0Fn[self.polyOrder-1](xC, q[j], q[j-1], q[j+1], dx)
      #end
    #end

    values = values[..., np.newaxis]
    if overwrite:
      self.data.push(grid, values)
    else:
      return grid, values
    #end
  #end
#end
