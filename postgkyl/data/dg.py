"""
Postgkyl sub-module to interpolate G* data
"""
import tables
import numpy
import os
from glob import glob
from postgkyl.data.computeInterpolationMatrices import createInterpMatrix
from postgkyl.data.computeDerivativeMatrices import createDerivativeMatrix

postgkylPath = os.path.dirname(os.path.realpath(__file__))

def _getNumNodes(dim, polyOrder, basis):
    numNodesSerendipity = numpy.array([[ 2,   3,   4,   5],
                                       [ 4,   8,  12,  17],
                                       [ 8,  20,  32,  50],
                                       [16,  48,  80, 136],
                                       [32, 112, 192, 352]])
    numNodesMaximal = numpy.array([[2,  3,  4,   5],
                                 [3,  6, 10,  15],
                                 [4, 10, 20,  35],
                                 [5, 15, 35,  70],
                                 [6, 21, 56, 126]])
    if basis.lower() == 'ns' or basis.lower() == 'ms':
        numNodes = numNodesSerendipity[dim-1, polyOrder-1]
    elif basis.lower() == 'mo':
        numNodes = numNodesMaximal[dim-1, polyOrder-1]
    else:
        raise NameError(
            "GInterp: Basis {} is not supported!\nSupported basis are currently 'ns' (Nodal Serendipity), 'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
            format(basis))
    return numNodes

def _getDataPath():
    path = os.path.dirname(os.path.realpath(__file__))
    # test if path contains h5 files
    test = glob(path + '/*.h5')
    if test == []:
        path = path + '/../../../../../data' #Conda path
        
    return path

def _loadInterpMatrix(dim, polyOrder, basis, interp, read):
    if interp is not None and read is None:
        mat = createInterpMatrix(dim, polyOrder, basis, interp)
        return mat
    elif read is not None:
        fileNameGeneral = postgkylPath + '/interpMatrix.h5'
        if not os.path.isfile(fileNameGeneral):
            print('interpMatrix.h5 not found, creating interpolation matrix\n')
            if interp is not None:
                print('interp = {}, calling createInterpMatrix for dim = {}, basis = {}, and polyOrder = {}\n'.format(interp, dim, basis, polyOrder))
                mat = createInterpMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print('interp not specified, reading from pre-computed files for dim = {}, basis = {}, and polyOrder = {}\n'.format(dim, basis, polyOrder))
                #Load interpolation matrix from the pre-computed HDF5 file since interp not specified.
                varid = 'xformMatrix%i%i' % (dim, polyOrder)
                if basis.lower() == 'ns':
                    fileName = _getDataPath() + '/xformMatricesNodalSerendipity.h5'

                elif basis.lower() == 'ms':
                    fileName = _getDataPath() + '/xformMatricesModalSerendipity.h5'

                elif basis.lower() == 'mo':
                    fileName = _getDataPath() + '/xformMatricesModalMaximal.h5'

                else:
                    raise NameError(
                        "GInterp: Basis {} is not supported!\nSupported basis are currently 'ns' (Nodal Serendipity), 'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
                        format(basis))
                fh = tables.open_file(fileName)
                mat = numpy.array(fh.root.matrices._v_children[varid].read())
                fh.close()
                return mat.transpose()
                
        fhGeneral = tables.open_file(postgkylPath + \
                              '/interpMatrix.h5', mode = 'r')        
        mat = fhGeneral.root.interpolation_matrix[:]
        fhGeneral.close()
        
        #Check if matrix loaded from interpMatrix.h5 has correct first dimension (i.e. has correct number of nodes)
        
        numNodes = _getNumNodes(dim, polyOrder, basis)
        if mat.shape[1] != numNodes:
            print('WARNING: Matrix read from interpMatrix.h5 does not have the right dimensions for specified dim = {}, basis = {}, and polyOrder = {}\n'.format(dim, basis, polyOrder))
            
            if interp is not None:
                print('interp = {}, calling createInterpMatrix for dim = {}, basis = {}, and polyOrder = {}\n'.format(interp, dim, basis, polyOrder))
                mat = createInterpMatrix(dim, polyOrder, basis, interp)
                return mat
            else:
                print('interp not specified, reading from pre-computed files for dim = {}, basis = {}, and polyOrder = {}\n'.format(dim, basis, polyOrder))
                #Load interpolation matrix from the pre-computed HDF5 file since interp not specified.
                varid = 'xformMatrix%i%i' % (dim, polyOrder)
                if basis.lower() == 'ns':
                    fileName = _getDataPath() + '/xformMatricesNodalSerendipity.h5'

                elif basis.lower() == 'ms':
                    fileName = _getDataPath() + '/xformMatricesModalSerendipity.h5'

                elif basis.lower() == 'mo':
                    fileName = _getDataPath() + '/xformMatricesModalMaximal.h5'
                    
                else:
                    raise NameError(
                        "GInterp: Basis {} is not supported!\nSupported basis are currently 'ns' (Nodal Serendipity), 'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
                        format(basis))
                fh = tables.open_file(fileName)
                mat = numpy.array(fh.root.matrices._v_children[varid].read())
                fh.close()
                return mat.transpose()
        return mat
    else:
        #Load interpolation matrix from the pre-computed HDF5 file.
        varid = 'xformMatrix%i%i' % (dim, polyOrder)
        if basis.lower() == 'ns':
            fileName = _getDataPath() + '/xformMatricesNodalSerendipity.h5'

        elif basis.lower() == 'ms':
            fileName = _getDataPath() + '/xformMatricesModalSerendipity.h5'

        elif basis.lower() == 'mo':
            fileName = _getDataPath() + '/xformMatricesModalMaximal.h5'
            
        else:
            raise NameError(
                "GInterp: Basis {} is not supported!\nSupported basis are currently 'ns' (Nodal Serendipity), 'ms' (Modal Serendipity), and 'mo' (Modal Maximal Order)".
                format(basis))
        fh = tables.open_file(fileName)
        mat = numpy.array(fh.root.matrices._v_children[varid].read())
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

        #Check if matrix loaded from interpMatrix.h5 has correct first dimension (i.e. has correct number of nodes)
        
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
    - GInterpNodal
    - GInterpModal
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

    def __init__(self, data):
        self.numDims = data.numDims
        GInterp.__init__(self, data, 1)

    def interpolate(self, c):
        return numpy.array(self.Xc), numpy.squeeze(self._getRawNodal(c))
    
    def differentiate(self, direction, comp=0):
        q = numpy.squeeze(self._getRawNodal(comp))
        coords = numpy.array(self.Xc)
        if direction is not None:
            return coords, np.gradient(q, coords[direction][1] - coords[direction][0], axis=direction, edge_order=2)
        else:
            derivativeData = np.zeros(q.shape, self.numDims)
            for i in range(0,self.numDims):
                derivativeData[:,i] = np.gradient(q, coords[i][1] - coords[i][0], axis=i, edge_order=2)
                derivativeData[:,i] /= (coords[i][1] - coords[i][0])
            return coords, derivativeData
        
class GInterpNodal(GInterp):
    """Class for manipulating nodal DG data
    """

    def __init__(self, data, polyOrder, basis, numInterp=None, read=None):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        self.basis = basis
        self.numInterp = numInterp
        self.read = read
        numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basis)
        GInterp.__init__(self, data,
                             numNodes)

    def interpolate(self, comp=0):
        q = self._getRawNodal(comp)
        cMat = _loadInterpMatrix(self.numDims, self.polyOrder, self.basis, self.numInterp, self.read)
        coords = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))), self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(cMat, q)

    def differentiate(self, direction, comp=0):
        q = self._getRawNodal(comp)
        cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder, self.basis, self.numInterp, self.read)
        coords = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))), self.Xc[d])
                  for d in range(self.numDims)]
        if direction is not None:
            return numpy.array(coords), _interpOnMesh(cMat[:,:,direction], q)/(self.Xc[direction][1]-self.Xc[direction][0])
        else:
            derivativeData = np.zeros(q.shape, self.numDims)
            for i in range(0,self.numDims):
                derivativeData[:,i] = _interpOnMesh(cMat[:,:,i], q)
                derivativeData[:,i] /= (self.Xc[i][1]-self.Xc[i][0])
            return numpy.array(coords), derivativeData

class GInterpModal(GInterp):
    """Class for manipulating modal DG data
    """

    def __init__(self, data, polyOrder, basis, numInterp=None, read=None):
        self.numDims = data.numDims
        self.polyOrder = polyOrder
        self.basis = basis
        self.numInterp = numInterp
        self.read = read
        numNodes = _getNumNodes(self.numDims, self.polyOrder, self.basis)
        GInterp.__init__(self, data,
                             numNodes)

    def interpolate(self, comp=0):
        q = self._getRawModal(comp)
        cMat = _loadInterpMatrix(self.numDims, self.polyOrder, self.basis, self.numInterp, self.read)
        coords = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))), self.Xc[d])
                  for d in range(self.numDims)]
        return numpy.array(coords), _interpOnMesh(cMat, q)

    def differentiate(self, direction, comp=0):
        q = self._getRawModal(comp)
        cMat = _loadDerivativeMatrix(self.numDims, self.polyOrder, self.basis, self.numInterp, self.read)
        coords = [_makeMesh(int(round(cMat.shape[0] ** (1.0/self.numDims))), self.Xc[d])
                  for d in range(self.numDims)]
        if direction is not None:
            return numpy.array(coords), _interpOnMesh(cMat[:,:,direction], q)/(self.Xc[direction][1]-self.Xc[direction][0])
        else:
            derivativeData = np.zeros(q.shape, self.numDims)
            for i in range(0,self.numDims):
                derivativeData[:,i] = _interpOnMesh(cMat[:,:,i], q)
                derivativeData[:,i] /= (self.Xc[i][1]-self.Xc[i][0])
            return numpy.array(coords), derivativeData
