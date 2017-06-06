#!/usr/bin/env python
"""
Postgkyl sub-module to interpolate G* data
"""
import tables
import numpy
import os
from computeInterpolationMatrices import createInterpMatrix

postgkylPath = os.path.dirname(os.path.realpath(__file__))


def _loadMatrix(dim, polyOrder, basis):
    """Load interpolation matrix from the pre-computed HDF5 file."""
    varid = 'xformMatrix%i%i' % (dim, polyOrder)
    if basis.lower() == 'nodal serendipity':
        fileName = postgkylPath + '/xformMatricesNodalSerendipity.h5'
        if not os.path.isfile(fileName):
            fileName = postgkylPath + \
                       '/../../../../../data/xformMatricesNodalSerendipity.h5'
    elif basis.lower() == 'modal serendipity':
        fileName = postgkylPath + '/xformMatricesModalSerendipity.h5'
        if not os.path.isfile(fileName):
            fileName = postgkylPath + \
                       '/../../../../../data/xformMatricesModalSerendipity.h5'
    elif basis.lower() == 'modal maximal order':
        fileName = postgkylPath + '/xformMatricesModalMaximal.h5'
        if not os.path.isfile(fileName):
            fileName = postgkylPath + \
                       '/../../../../../data/xformMatricesModalMaximal.h5'
    else:
        raise NameError(
            "GInterp: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".
            format(basis))
    fh = tables.open_file(fileName)
    mat = numpy.array(fh.root.matrices._v_children[varid].read())
    fh.close()
    return mat


def _decompose(n, dim, numInterp):
    """Decompose n to the number decription with basis numInterp"""
    return numpy.mod(numpy.full(dim, n, dtype=numpy.int) /
                     (numInterp**numpy.arange(dim)), numInterp)


def _makeMesh(nInterp, Xc):
    dx = Xc[1] - Xc[0]
    nx = Xc.shape[0]
    xlo = Xc[0] - 0.5*dx
    xup = Xc[-1] + 0.5*dx
    dx2 = dx/nInterp
    return numpy.linspace(xlo+0.5*dx2, xup-0.5*dx2, nInterp*nx)


def _interpOnMesh(cMat, qIn):
    """Interpolate DG data on a uniform mesh.

    Parameters:
    cMat -- interpolation matrix (loaded using loadMatrix)
    qIn  -- data to project
    """
    numCells = numpy.array(qIn.shape)
    # last entry is indexing nodes, get rid of it
    numCells = numCells[:-1]
    numDims = int(len(numCells))
    numInterp = int(round(cMat.shape[0] ** (1.0/numDims)))
    numNodes = cMat.shape[1]
    qOut = numpy.zeros(numCells*numInterp, numpy.float)
    # move the node index from last to the first
    qIn = numpy.moveaxis(qIn, -1, 0)

    # Main loop
    for n in range(numInterp ** numDims):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        temp = numpy.tensordot(cMat[n, :], qIn, axes=1)
        # decompose n to i,j,k,... indices based on the number of dimensions
        startIdx = _decompose(n, numDims, numInterp)
        # define multi-D qOut slices
        idxs = [slice(int(startIdx[i]), int(numCells[i]*numInterp), numInterp)
                for i in range(numDims)]
        qOut[idxs] = temp
    return numpy.array(qOut)


class GInterp:
    """Base class for DG interpolation.

    __init__(data : GData, numNodes : int)

    Note:
    This class should not be used on its own. Currently supported
    child classes are:
    - GInterpZeroOrder
    - GInterpNodalSerendipity
    - GInterpModalMaxOrder
    """

    def __init__(self, dat, numNodes):
        self.q = dat.q
        self.numNodes = numNodes
        self.numEqns = dat.q.shape[-1]/numNodes
        self.numDims = dat.numDims
        self.dx = (dat.upperBounds - dat.lowerBounds)/dat.numCells
        self.Xc = [numpy.linspace(dat.lowerBounds[d] + 0.5*self.dx[d],
                                  dat.upperBounds[d] - 0.5*self.dx[d],
                                  dat.numCells[d])
                   for d in range(self.numDims)]

    def _getRawNodal(self, component):
        q = self.q
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.numDims)]
        shp.append(self.numNodes)
        rawData = numpy.zeros(shp, numpy.float)
        for n in range(self.numNodes):
            rawData[..., n] = q[..., int(component+n*numEqns)]
        return rawData

    def _getRawModal(self, component):
        q = self.q
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.numDims)]
        shp.append(self.numNodes)
        rawData = numpy.zeros(shp, numpy.float)
        lo = int(component*self.numNodes)
        up = int(lo+self.numNodes)
        rawData = q[..., lo:up]
        return rawData


class GInterpZeroOrder(GInterp):
    """This is provided to allow treating finite-volume data as DG
    with piecewise constant basis.
    """

    def __init__(self, dat):
        GInterp.__init__(self, dat, 1)

    def project(self, c):
        return numpy.array(self.Xc), numpy.squeeze(self._getRawNodal(c))


class GInterpNodalSerendipity(GInterp):
    """Nodal Serendipity basis PUT MORE STUF HERE"""

    numNodes = numpy.array([[ 2,   3,   4,   5],
                            [ 4,   8,  12,  17],
                            [ 8,  20,  32,  50],
                            [16,  48,  80, 136],
                            [32, 112, 192, 352]])

    def __init__(self, data, polyOrder):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        GInterp.__init__(self, data,
                         self.numNodes[self.numDims-1, self.polyOrder-1])
        self.cMat = _loadMatrix(self.numDims, self.polyOrder,
                                'nodal Serendipity')

    def project(self, comp=0):
        q = self._getRawNodal(comp)
        coords = [_makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(self.cMat.transpose(), q)


class GInterpModalSerendipity(GInterp):
    """Modal Serendipity basis PUT MORE STUF HERE"""

    numNodes = numpy.array([[ 2,   3,   4,   5],
                            [ 4,   8,  12,  17],
                            [ 8,  20,  32,  50],
                            [16,  48,  80, 136],
                            [32, 112, 192, 352]])

    def __init__(self, data, polyOrder):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        GInterp.__init__(self, data,
                         self.numNodes[self.numDims-1, polyOrder-1])
        self.cMat = _loadMatrix(self.numDims, self.polyOrder,
                                'modal Serendipity')

    def project(self, comp=0):
        q = self._getRawModal(comp)
        coords = [_makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(self.cMat.transpose(), q)


class GInterpModalMaxOrder(GInterp):
    """Modal Maximal Order basis PUT MORE STUF HERE"""

    numNodes = numpy.array([[2,  3,  4,   5],
                            [3,  6, 10,  15],
                            [4, 10, 20,  35],
                            [5, 15, 35,  70],
                            [6, 21, 56, 126]])

    def __init__(self, data, polyOrder):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        GInterp.__init__(self, data,
                         self.numNodes[self.numDims-1, polyOrder-1])
        self.cMat = _loadMatrix(self.numDims, self.polyOrder,
                                'modal Maximal Order')

    def project(self, comp=0):
        q = self._getRawModal(comp)
        coords = [_makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(self.cMat.transpose(), q)

class GInterpGeneral(GInterp):
    """General interpolation routine - calls createInterpMatrix to generate an interpolation matrix for arbitrary level of refinement"""

    def __init__(self, data, polyOrder, basis, numInterp):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        self.basis = basis
        self.numInterp = numInterp
        self.cMat = createInterpMatrix(self.numDims, self.polyOrder, self.basis, self.numInterp)
        GInterp.__init__(self, data,
                         self.cMat.shape[1])

    def project(self, comp=0):
        if self.basis == 'ns':
            q = self._getRawNodal(comp)
        else:
            q = self._getRawModal(comp)
        coords = [_makeMesh(self.numInterp, self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(self.cMat, q)

class GInterpGeneralRead(GInterp):
    """General interpolation routine - reads interpMatrix.h5 which contains an interpolation matrix for arbitrary level of refinement"""
    numNodesSerendipity = numpy.array([[ 2,   3,   4,   5],
                                       [ 4,   8,  12,  17],
                                       [ 8,  20,  32,  50],
                                       [16,  48,  80, 136],
                                       [32, 112, 192, 352]])
    numNodesModal = numpy.array([[2,  3,  4,   5],
                                 [3,  6, 10,  15],
                                 [4, 10, 20,  35],
                                 [5, 15, 35,  70],
                                 [6, 21, 56, 126]])    
    
    def __init__(self, data, polyOrder, basis):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        self.basis = basis
        fh = tables.open_file(postgkylPath + \
                       '/interpMatrix.h5', mode = 'r')
        self.cMat = fh.root.interpolation_matrix[:]
        if basis == 'mo' and self.cMat.shape[1] != self.numNodesModal[self.numDims-1, self.polyOrder-1]:
            raise NameError("interp: interpMatrix from file is not the right size to interpolate the given data")
        elif (basis == 'ms' or basis == 'ns') and self.cMat.shape[1] != self.numNodesSerendipity[self.numDims-1, self.polyOrder-1]:
            raise NameError("interp: interpMatrix from file is not the right size to interpolate the given data")
        else:
            GInterp.__init__(self, data,
                             self.cMat.shape[1])

    def project(self, comp=0):
        if self.basis == 'ns':
            q = self._getRawNodal(comp)
        else:
            q = self._getRawModal(comp)
        numInterp = int(round(self.cMat.shape[0] ** (1.0/self.numDims)))
        coords = [_makeMesh(numInterp, self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(self.cMat, q)    
