#!/usr/bin/env python
"""
Postgkyl module to interpolate G* data
"""

import tables
import gData
import numpy
import os
import exceptions

# obtain Postgkyl path
postgkylPath = os.path.dirname(os.path.realpath(__file__))

## Below are a set of helper functions used in the DG classes

def loadMatrix(dim, polyOrder, basis):
    """Load interpolation matrix from the pre-computed HDF5 file."""
    varid ='xformMatrix%i%i' % (dim,polyOrder)
    if basis.lower() == 'nodal serendipity':
        fh = tables.open_file(postgkylPath+
                              '/xformMatricesNodalSerendipity.h5')
        mat = numpy.array(fh.root.matrices._v_children[varid].read())
    elif basis.lower() == 'modal serendipity':
        fh = tables.open_file(postgkylPath+
                              '/xformMatricesModalSerendipity.h5')
        mat = numpy.array(fh.root.matrices._v_children[varid].read())
    elif basis.lower() == 'modal maximal order':
        fh = tables.open_file(postgkylPath+
                              '/xformMatricesModalMaximal.h5')
        mat = numpy.array(fh.root.matrices._v_children[varid].read())
    else:
        raise exceptions.RuntimeError(
            "GInterp: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".
            format(basis))
    fh.close()
    return mat    

def decompose(n, dim, numInterp):
    """Decompose n to the number decription with basis numInterp"""
    return numpy.mod( numpy.full(dim, n, dtype=numpy.int) / 
                      (numInterp**numpy.arange(dim)), numInterp )

def makeMesh(nInterp, Xc):
    dx = Xc[1] - Xc[0]
    nx = Xc.shape[0]
    xlo = Xc[0]  - 0.5*dx
    xup = Xc[-1] + 0.5*dx
    dx2 = dx/nInterp
    return numpy.linspace(xlo+0.5*dx2, xup-0.5*dx2, nInterp*nx)

def interpOnMesh(cMat, qIn):
    """Interpolate DG data on a uniform mesh.

    Parameters:
    cMat -- interpolation matrix (loaded using loadMatrix)
    qIn  -- data to project
    """
    numCells = numpy.array(qIn.shape)
    # last entry is indexing nodes, get rid of it
    numCells = numCells[:-1]
    numDims = len(numCells)
    numInterp = int(cMat.shape[0] ** (1.0/numDims))
    numNodes = cMat.shape[1]
    qOut = numpy.zeros(numCells*numInterp, numpy.float)
    # move the node index from last to the first
    qIn = numpy.moveaxis(qIn, -1, 0)

    # Main loop
    for n in range(numInterp ** numDims):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        temp  = numpy.tensordot(cMat[n, :], qIn, axes=1)
        # decompose n to i,j,k,... indices based on the number of dimensions
        startIdx = decompose(n, numDims, numInterp)
        # define multi-D qOut slices
        idxs = [slice(startIdx[i], numCells[i]*numInterp, numInterp) 
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
            rawData[..., n] = q[..., component+n*numEqns]
        return rawData

    def _getRawModal(self, component):
        q = self.q
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.numDims)]
        shp.append(self.numNodes)
        rawData = numpy.zeros(shp, numpy.float)
        lo = component*self.numNodes
        up = lo+self.numNodes
        rawData = q[..., lo:up]
        return rawData    

#################

class GInterpZeroOrder(GInterp):
    """This is provided to allow treating finite-volume data as DG
    with piecewise constant basis.
    """

    def __init__(self, dat):
        GInterp.__init__(self, dat, 1)

    def project(self, c):
        return self.Xc[0], self._getRawNodal(c)

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
                         self.numNodes[self.numDims-1, polyOrder-1])
        self.cMat = loadMatrix(self.numDims, self.polyOrder,
                               'nodal Serendipity')

    def project(self, comp=0):
        q = self._getRawNodal(comp)
        coords = [makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        grids = numpy.meshgrid(*coords, indexing='ij')
        return numpy.array(grids), interpOnMesh(self.cMat.transpose(), q)

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
        self.cMat = loadMatrix(self.numDims, self.polyOrder,
                               'modal Serendipity')

    def project(self, comp=0):
        q = self._getRawModal(comp)
        coords = [makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        grids = numpy.meshgrid(*coords, indexing='ij')
        return numpy.array(grids), interpOnMesh(self.cMat.transpose(), q)

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
        self.cMat = loadMatrix(self.numDims, self.polyOrder,
                               'modal Maximal Order')

    def project(self, comp=0):
        q = self._getRawModal(comp)
        coords = [makeMesh(self.polyOrder+1, self.Xc[d])
                  for d in range(self.numDims)]
        grids = numpy.meshgrid(*coords, indexing='ij')
        return numpy.array(grids), interpOnMesh(self.cMat.transpose(), q)
