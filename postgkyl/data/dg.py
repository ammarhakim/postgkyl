import os.path
from glob import glob

import tables
import numpy as np

from postgkyl.data.gdata import GData
from postgkyl.data.computeInterpolationMatrices import createInterpMatrix
from postgkyl.data.computeDerivativeMatrices import createDerivativeMatrix

path = os.path.dirname(os.path.realpath(__file__))

def _getNumNodes(dim, polyOrder, basis):
    if basis.lower() == 'ns' or basis.lower() == 'ms':
        numNodesSerendipity = np.array([[ 2,   3,   4,   5],
                                        [ 4,   8,  12,  17],
                                        [ 8,  20,  32,  50],
                                        [16,  48,  80, 136],
                                        [32, 112, 192, 352]])
        numNodes = numNodesSerendipity[dim-1, polyOrder-1]
    elif basis.lower() == 'mo':
        numNodesMaximal = np.array([[2,  3,  4,   5],
                                    [3,  6, 10,  15],
                                    [4, 10, 20,  35],
                                    [5, 15, 35,  70],
                                    [6, 21, 56, 126]])
        numNodes = numNodesMaximal[dim-1, polyOrder-1]
    else:
        raise NameError(
            "GInterp: Basis '{:s}' is not supported!\n"
            "Supported basis are currently 'ns' (Nodal Serendipity),"
            " 'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
            format(basis))
    return numNodes

def _loadInterpMatrix(dim, polyOrder, basis, interp, read):
    if interp is not None and read is None:
        mat = createInterpMatrix(dim, polyOrder, basis, interp)
        return mat
    elif read is not None:
        fileNameGeneral = postgkylPath + '/interpMatrix.h5'
        if not os.path.isfile(fileNameGeneral):
            print("interpMatrix.h5 not found, creating interpolation matrix\n")
            if interp is not None:
                print("interp = {}, calling createInterpMatrix for dim = {},"
                      " basis = {}, and polyOrder = {}\n".
                      format(interp, dim, basis, polyOrder))
                mat = createInterpMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print("interp not specified, reading from pre-computed"
                      " files for dim = {}, basis = {}, and polyOrder = {}\n".
                      format(dim, basis, polyOrder))
                # Load interpolation matrix from the pre-computed HDF5
                # file since interp not specified.
                varid = "xformMatrix{:d}{:d}".format(dim, polyOrder)
                if basis.lower() == 'ns':
                    fileName = path \
                               + '/xformMatricesNodalSerendipity.h5'
                elif basis.lower() == 'ms':
                    fileName = path \
                               + '/xformMatricesModalSerendipity.h5'
                elif basis.lower() == 'mo':
                    fileName = path \
                               + '/xformMatricesModalMaximal.h5'

                else:
                    raise NameError(
                        "GInterp: Basis {} is not supported!\n"
                        "Supported basis are currently 'ns'"
                        " (Nodal Serendipity), 'ms' (Modal Serendipity),"
                        " and 'mo' (Modal Maximal Order)".
                        format(basis))
                fh = tables.open_file(fileName)
                mat = np.array(fh.root.matrices._v_children[varid].read())
                fh.close()
                return mat.transpose()
                
        fhGeneral = tables.open_file(postgkylPath + \
                              '/interpMatrix.h5', mode = 'r')        
        mat = fhGeneral.root.interpolation_matrix[:]
        fhGeneral.close()
        
        # Check if matrix loaded from interpMatrix.h5 has correct first
        # dimension (i.e. has correct number of nodes)
        
        numNodes = _getNumNodes(dim, polyOrder, basis)
        if mat.shape[1] != numNodes:
            print("WARNING: Matrix read from interpMatrix.h5"
                  " does not have the right dimensions for specified "
                  "dim = {:d}, basis = {:d}, and polyOrder = {:d}\n".
                  format(dim, basis, polyOrder))
            
            if interp is not None:
                print("interp = {}, calling createInterpMatrix "
                "for dim = {:d}, basis = {:d}, and polyOrder = {:d}\n".
                      format(interp, dim, basis, polyOrder))
                mat = createInterpMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print("interp not specified, reading from pre-computed "
                      "files for dim = {:d}, basis = {:d}, "
                      "and polyOrder = {}\n".
                      format(dim, basis, polyOrder))
                # Load interpolation matrix from the pre-computed HDF5
                # file since interp not specified.
                varid = "xformMatrix{:d}{:d}".format(dim, polyOrder)
                if basis.lower() == 'ns':
                    fileName = path \
                               + '/xformMatricesNodalSerendipity.h5'
                elif basis.lower() == 'ms':
                    fileName = path \
                               + '/xformMatricesModalSerendipity.h5'
                elif basis.lower() == 'mo':
                    fileName = path \
                               + '/xformMatricesModalMaximal.h5'
                else:
                    raise NameError(
                        "GInterp: Basis {} is not supported!\n"
                        "Supported basis are currently 'ns' "
                        "(Nodal Serendipity), 'ms' (Modal Serendipity), "
                        "and 'mo' (Modal Maximal Order)".
                        format(basis))
                fh = tables.open_file(fileName)
                mat = fh.root.matrices._v_children[varid].read()
                fh.close()
                return mat.transpose()
        return mat
    else:
        # Load interpolation matrix from the pre-computed HDF5 file.
        varid = 'xformMatrix%i%i' % (dim, polyOrder)
        if basis.lower() == 'ns':
            fileName = path + '/xformMatricesNodalSerendipity.h5'

        elif basis.lower() == 'ms':
            fileName = path + '/xformMatricesModalSerendipity.h5'

        elif basis.lower() == 'mo':
            fileName = path + '/xformMatricesModalMaximal.h5'
            
        else:
            raise NameError(
                "GInterp: Basis {:s} is not supported!\n"
                "Supported basis are currently 'ns' (Nodal Serendipity), "
                "'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
                format(basis))
        fh = tables.open_file(fileName)
        mat = fh.root.matrices._v_children[varid].read()
        fh.close()
        return mat.transpose()

def _loadDerivativeMatrix(dim, polyOrder, basis, interp, read):
    if interp is not None and read is None:
        mat = createDerivativeMatrix(dim, polyOrder, basis, interp)
        return mat
    elif read is not None:
        fileNameGeneral = postgkylPath + '/derivativeMatrix.h5'
        if not os.path.isfile(fileNameGeneral):
            print('derivativeMatrix.h5 not found, creating derivative matrix\n')
            if interp is not None:
                print('interp = {}, calling createDerivativeMatrix for dim = {}, basis = {}, and polyOrder = {}\n'.format(interp, dim, basis, polyOrder))
                mat = createDerivativeMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print('interp not specified, calling createDerivativeMatrix with polyOrder = {}+1 level of interpolation\n'.format(polyOrder))
                mat = createDerivativeMatrix(dim, polyOrder, basis, polyOrder+1)
                return mat

        fhGeneral = tables.open_file(postgkylPath + \
                              '/derivativeMatrix.h5', mode = 'r')        
        mat = fhGeneral.root.derivative_matrix[:]
        fhGeneral.close()

        # Check if matrix loaded from interpMatrix.h5 has correct
        # first dimension (i.e. has correct number of nodes)
        
        numNodes = _getNumNodes(dim, polyOrder, basis)
        if mat.shape[1] != numNodes:
            print('WARNING: Matrix read from derivativeMatrix.h5 does not have the right dimensions for specified dim = {}, basis = {}, and polyOrder = {}\n'.format(dim, basis, polyOrder))
            
            if interp is not None:
                print('interp = {}, calling createDerivativeMatrix for dim = {}, basis = {}, and polyOrder = {}\n'.format(interp, dim, basis, polyOrder))
                mat = createDerivativeMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print('interp not specified, calling createDerivativeMatrix with polyOrder = {}+1 level of interpolation\n'.format(polyOrder))
                mat = createDerivativeMatrix(dim, polyOrder, basis, polyOrder+1)
                return mat

        return mat
    else:
        interp = polyOrder+1
        mat = createDerivativeMatrix(dim, polyOrder, basis, interp)
        return mat
        
def _decompose(n, dim, numInterp):
    """Decompose n to the number decription with basis numInterp"""
    return np.mod(np.full(dim, n, dtype=np.int) /
                     (numInterp**np.arange(dim)), numInterp)


def _makeMesh(nInterp, Xc, xlo=None, xup=None):
    nx = Xc.shape[0]
    if xlo is None or xup is None:
        if nx == 1:
            raise ValueError("Cannot create interpolated grid from 1 cell without specifying 'xlo' and 'xup'")
        else:
            dx = (Xc[-1] - Xc[0]) / (nx-1)
        xlo = Xc[0] - 0.5*dx
        xup = Xc[-1] + 0.5*dx
    else:
        dx = (xup - xlo) / nx
    dx2 = dx/nInterp
    return np.linspace(xlo+0.5*dx2, xup-0.5*dx2, nInterp*nx)


def _interpOnMesh(cMat, qIn):
    numCells = np.array(qIn.shape)
    # last entry is indexing nodes, get rid of it
    numCells = numCells[:-1]
    numDims = int(len(numCells))
    numInterp = int(round(cMat.shape[0] ** (1.0/numDims)))
    numNodes = cMat.shape[1]
    qOut = np.zeros(numCells*numInterp, np.float)
    # move the node index from last to the first
    qIn = np.moveaxis(qIn, -1, 0)

    # Main loop
    for n in range(numInterp ** numDims):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        temp = np.tensordot(cMat[n, :], qIn, axes=1)
        # decompose n to i,j,k,... indices based on the number of dimensions
        startIdx = _decompose(n, numDims, numInterp)
        # define multi-D qOut slices
        idxs = [slice(int(startIdx[i]), int(numCells[i]*numInterp), numInterp)
                for i in range(numDims)]
        qOut[idxs] = temp
    return np.array(qOut)

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
        lower, upper = data.getBounds()
        cells = data.getNumCells()
        self.dx = (upper - lower)/cells
        grid = data.peakGrid()
        xlo, xup = data.getBounds()
        self.Xc = grid
        self.xlo = xlo
        self.xup = xup

    def _getRawNodal(self, component):
        q = self.data.peakValues()
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.numDims)]
        shp.append(self.numNodes)
        rawData = np.zeros(shp, np.float)
        for n in range(self.numNodes):
            rawData[..., n] = q[..., int(component+n*numEqns)]
        return rawData

    def _getRawModal(self, component):
        q = self.data.peakValues()
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.numDims)]
        shp.append(self.numNodes)
        rawData = np.zeros(shp, np.float)
        lo = int(component*self.numNodes)
        up = int(lo+self.numNodes)
        rawData = q[..., lo:up]
        return rawData
        
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

    def __init__(self, data, polyOrder, basis,
                 numInterp=None, read=None):
        self.numDims = data.getNumDims()
        self.polyOrder = polyOrder
        self.basis = basis
        self.numInterp = numInterp
        self.read = read
        numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basis)
        GInterp.__init__(self, data, numNodes)

    def interpolate(self, comp=0, stack=False):
        q = self._getRawNodal(comp)
        cMat = _loadInterpMatrix(self.numDims, self.polyOrder,
                                 self.basis, self.numInterp, self.read)
        grid = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))),
                          self.Xc[d], xlo=self.xlo[d], xup=self.xup[d])
                  for d in range(self.numDims)]
        values = _interpOnMesh(cMat, q)[..., np.newaxis]

        if stack is False:
            return grid, values
        else:
            self.data.pushGrid(grid)
            self.data.pushValues(values)

    def differentiate(self, direction, comp=0, stack=False):
        q = self._getRawNodal(comp)
        cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder,
                                     self.basis, self.numInterp, self.read)
        grid = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))), 
                          self.Xc[d], xlo=self.xlo[d], xup=self.xup[d])
                  for d in range(self.numDims)]
        if direction is not None:
            values = _interpOnMesh(cMat[:, :, direction], q) /\
                     (self.Xc[direction][1] - self.Xc[direction][0])
        else:
            values = np.zeros(q.shape, self.numDims)
            for i in range(self.numDims):
                values[:,i] = _interpOnMesh(cMat[:,:,i], q)
                values[:,i] /= (self.Xc[i][1]-self.Xc[i][0])
        values = values[..., np.newaxis]

        if stack is False:
            return grid, values
        else:
            self.data.pushGrid(grid)
            self.data.pushValues(values)

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

    def __init__(self, data, polyOrder, basis,
                 numInterp=None, read=None):
        self.numDims = data.getNumDims()
        self.polyOrder = polyOrder
        self.basis = basis
        self.numInterp = numInterp
        self.read = read
        numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basis)
        GInterp.__init__(self, data, numNodes)

    def interpolate(self, comp=0, stack=False):
        q = self._getRawModal(comp)
        cMat = _loadInterpMatrix(self.numDims, self.polyOrder,
                                 self.basis, self.numInterp, self.read)
        grid = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))),
                          self.Xc[d], xlo=self.xlo[d], xup=self.xup[d])
                  for d in range(self.numDims)]
        values = _interpOnMesh(cMat, q)[..., np.newaxis]

        if stack is False:
            return grid, values
        else:
            self.data.pushGrid(grid)
            self.data.pushValues(values)

    def differentiate(self, direction, comp=0, stack=False):
        q = self._getRawModal(comp)
        cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder,
                                     self.basis, self.numInterp, self.read)
        grid = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))),
                          self.Xc[d], xlo=self.xlo[d], xup=self.xup[d])
                  for d in range(self.numDims)]
        if direction is not None:
            values = _interpOnMesh(cMat[:, :, direction], q) /\
                     (self.Xc[direction][1] - self.Xc[direction][0])
        else:
            values = np.zeros(q.shape, self.numDims)
            for i in range(self.numDims):
                values[:,i] = _interpOnMesh(cMat[:,:,i], q)
                values[:,i] /= (self.Xc[i][1]-self.Xc[i][0])
        values = values[..., np.newaxis]

        if stack is False:
            return grid, values
        else:
            self.data.pushGrid(grid)
            self.data.pushValues(values)

# class GInterpZeroOrder(GInterp):
#     """This is provided to allow treating finite-volume data as DG
#     with piecewise constant basis.
#     """

#     def __init__(self, data):
#         self.numDims = data.getNumDims()
#         GInterp.__init__(self, data, 1)

#     def interpolate(self, c):
#         return np.array(self.Xc),
#         np.squeeze(self._getRawNodal(c))[..., np.newaxis]
    
#     def differentiate(self, direction, comp=0):
#         q = np.squeeze(self._getRawNodal(comp))
#         grid = np.array(self.Xc)
#         if direction is not None:
#             return grid, np.gradient(q, coords[direction][1] - coords[direction][0], axis=direction, edge_order=2)
#         else:
#             derivativeData = np.zeros(q.shape, self.numDims)
#             for i in range(0,self.numDims):
#                 derivativeData[:,i] = np.gradient(q, coords[i][1] - coords[i][0], axis=i, edge_order=2)
#                 derivativeData[:,i] /= (coords[i][1] - coords[i][0])
#             return grid, derivativeData[..., np.newaxis]
