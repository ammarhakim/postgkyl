import gkedata
import gkedginterpdat as gid
import pylab
import numpy
import math

## Below are a set of helper functions used in the DG classes

def makeMatrix(*ll):
    nrow = len(ll)
    ncol = len(ll[0])
    mat = numpy.zeros((nrow,ncol), numpy.float)
    for i in range(nrow):
        mat[i,:] = ll[i]
    return mat

def evalSum(coeff, fields):
    res = 0.0*fields[0]
    for i in range(len(coeff)):
        res = res + coeff[i]*fields[i]
    return res

def makeMesh(nInterp, Xc):
    dx = Xc[1]-Xc[0]
    nx = Xc.shape[0]
    xlo = Xc[0]-0.5*dx
    xup = Xc[-1]+0.5*dx
    dx2 = dx/nInterp
    return pylab.linspace(xlo+0.5*dx2, xup-0.5*dx2, nInterp*nx)

def makeMesh2(nInterp, Xc):
    dx = Xc[1]-Xc[0]
    nx = Xc.shape[0]
    xlo = Xc[0]-0.5*dx
    xup = Xc[-1]+0.5*dx
    dx2 = dx/nInterp
    return pylab.linspace(xlo, xup, nInterp*nx+1)

def interpOnMesh1D(cMat, qIn):
    nInterp, nNodes = cMat.shape[0], cMat.shape[1]
    nx = qIn.shape[0]
    qout = pylab.zeros((nInterp*nx,), numpy.float)
    vList = [qIn[:,i] for i in range(nNodes)]
    for i in range(nInterp):
        qout[i:nInterp*nx:nInterp] = evalSum(cMat[i,:], vList)
    return qout

def interpOnMesh2D(cMat, qIn):
    nInterp, nNodes = int(math.sqrt(cMat.shape[0])), cMat.shape[1]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    qout = pylab.zeros((nInterp*nx,nInterp*ny), numpy.float)
    vList = [qIn[:,:,i] for i in range(nNodes)]
    n = 0
    for j in range(nInterp):
        for i in range(nInterp):
            qout[i:nInterp*nx:nInterp, j:nInterp*ny:nInterp] = evalSum(cMat[n,:], vList)
            n = n+1
    return qout

def interpOnMesh3D(cMat, qIn):
    nInterp, nNodes = int(cMat.shape[0] ** (1.0/3.0)), cMat.shape[1]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    nz = qIn.shape[2]
    qout = pylab.zeros((nInterp*nx,nInterp*ny,nInterp*nz), numpy.float)
    vList = [qIn[:,:,:,i] for i in range(nNodes)]
    n = 0
    for k in range(nInterp):
        for j in range(nInterp):
            for i in range(nInterp):
                qout[i:nInterp*nx:nInterp, j:nInterp*ny:nInterp, k:nInterp*nz:nInterp] = evalSum(cMat[n,:], vList)
                n = n+1
    return qout

def interpOnMesh4D(cMat, qIn):
    nInterp, nNodes = int(cMat.shape[0] ** (1.0/4.0)), cMat.shape[1]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    nz = qIn.shape[2]
    nv = qIn.shape[3]
    qout = pylab.zeros((nInterp*nx,nInterp*ny,nInterp*nz,nInterp*nv), numpy.float)
    vList = [qIn[:,:,:,:,i] for i in range(nNodes)]
    n = 0
    for l in range(nInterp):
        for k in range(nInterp):
            for j in range(nInterp):
                for i in range(nInterp):
                    qout[i:nInterp*nx:nInterp, j:nInterp*ny:nInterp, k:nInterp*nz:nInterp, l:nInterp*nv:nInterp] = evalSum(cMat[n,:], vList)
                    n = n+1
    return qout

def interpOnMesh5D(cMat, qIn):
    nInterp, nNodes = int(cMat.shape[0] ** (1.0/5.0)), cMat.shape[1]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    nz = qIn.shape[2]
    nv = qIn.shape[3]
    nu = qIn.shape[4]
    qout = pylab.zeros((nInterp*nx,nInterp*ny,nInterp*nz,nInterp*nv,nInterp*nu), numpy.float)
    vList = [qIn[:,:,:,:,:,i] for i in range(nNodes)]
    n = 0
    for m in range(nInterp):
        for l in range(nInterp):
            for k in range(nInterp):
                for j in range(nInterp):
                    for i in range(nInterp):
                        qout[i:nInterp*nx:nInterp, j:nInterp*ny:nInterp, k:nInterp*nz:nInterp, l:nInterp*nv:nInterp, m:nInterp*nu:nInterp] = evalSum(cMat[n,:], vList)
                        n = n+1
    return qout

def computeIntegratedQuantity1D(weights, qIn, Xc):
    nNodes = weights.shape[0]
    nx = qIn.shape[0]
    dx = Xc[1]-Xc[0]
    qout = 0.0
    for j in range(nx):
        for i in range(nNodes):
            qout = qout + 0.5*dx*weights[i]*qIn[j, i]
    return qout

def computeIntegratedQuantity2D(weights, qIn, Xc, Yc):
    nNodes = weights.shape[0]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    dx = Xc[1]-Xc[0]
    dy = Yc[1]-Yc[0]
    qout = 0.0
    for k in range(nx):
        for j in range(ny):
            for i in range(nNodes):
                qout = qout + 0.5*dx*dy*weights[i]*qIn[k, j, i]
    return qout

def computeIntegratedQuantity3D(weights, qIn, Xc, Yc, Zc):
    nNodes = weights.shape[0]
    nx = qIn.shape[0]
    ny = qIn.shape[1]
    nz = qIn.shape[2]
    dx = Xc[1]-Xc[0]
    dy = Yc[1]-Yc[0]
    dz = Zc[1]-Zc[0]
    qout = 0.0
    for l in range(nx):
        for k in range(ny):
            for j in range(nz):
                for i in range(nNodes):
                    qout = qout + 0.5*dx*dy*dz*weights[i]*qIn[l, k, j, i]
    return qout

# def computeIntegratedQuantity4D(weights, qIn):
#     nNodes = weights.shape[0]
#     nx = qIn.shape[0]
#     ny = qIn.shape[1]
#     nz = qIn.shape[2]
#     nv = qIn.shape[3]
#     qout = 0.0
#     for m in range(nx):
#         for l in range(ny):
#             for k in range(nz):
#                 for j in range(nv):
#                     for i in range(nNodes):
#                         qout = qout +  weights[i]*qIn[m, l, k, j, i]
#     return qout

# def computeIntegratedQuantity5D(weights, qIn):
#     nNodes = weights.shape[0]
#     nx = qIn.shape[0]
#     ny = qIn.shape[1]
#     nz = qIn.shape[2]
#     nv = qIn.shape[3]
#     nu = qIn.shape[4]
#     qout = 0.0
#     for n in range(nx):
#         for m in range(ny):
#             for l in range(nz):
#                 for k in range(nv):
#                     for j in range(nu):
#                         for i in range(nNodes):
#                             qout = qout +  weights[i]*qIn[n, m, l, k, j, i]
#     return qout

class GkeDgBasis:
    r"""__init__(dat : GkeData, numNodes : int) -> GkeDgData

    Base class for post-processing DG data. The derived class should
    set the number of nodes in the element.
    """

    def __init__(self, dat, numNodes):
        self.q = dat.q
        self.numNodes = numNodes
        self.numEqns = dat.q.shape[-1]/numNodes
        self.ndim = dat.upperBounds.shape[0]
        self.dx = (dat.upperBounds[:]-dat.lowerBounds[:])/dat.cells[:]
        self.Xc = []
        for d in range(self.ndim):
            self.Xc.append(
                pylab.linspace(dat.lowerBounds[d]+0.5*self.dx[d], dat.upperBounds[d]-0.5*self.dx[d], dat.cells[d])
                )

    def _evalSum(self, coeff, fields):
        r"""evalSum(coeff: [] float, fields: [] array)
        
        Sum arrays in 'fields' list, weighing them with values in 'coeff'
        list.
        """
        return evalSum(coeff, fields)

    def _getRaw(self, component):
        q = self.q
        numEqns = self.numEqns
        shp = [q.shape[i] for i in range(self.ndim)]
        shp.append(self.numNodes)
        rawData = numpy.zeros(shp, numpy.float)
        for n in range(self.numNodes):
            # THERE MUST BE A BETTER WAY TO DO THIS
            if self.ndim == 1:
                rawData[:,n] = q[:,component+n*numEqns]
            elif self.ndim == 2:
                rawData[:,:,n] = q[:,:,component+n*numEqns]
            elif self.ndim == 3:
                rawData[:,:,:,n] = q[:,:,:,component+n*numEqns]
            elif self.ndim == 4:
                rawData[:,:,:,:,n] = q[:,:,:,:,component+n*numEqns]
            elif self.ndim == 5:
                rawData[:,:,:,:,:,n] = q[:,:,:,:,:,component+n*numEqns]
            elif self.ndim == 6:
                rawData[:,:,:,:,:,:,n] = q[:,:,:,:,:,:,component+n*numEqns]

        return rawData        

    def project(self, c):
        r"""project(c : int)
        """
        return 0, 0

#################
class GkeDgPolyOrder0Basis(GkeDgBasis):
    r"""This is provided to allow treating finite-volume data as DG
    with piecwise contant basis.
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 1)

    def project(self, c):
        return self.Xc[0], self._getRaw(c)

#################
class GkeDgLobatto1DPolyOrder1Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 1 basis, in 1D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 2)
        self.cMat_i2 = gid.GkeDgLobatto1DPolyOrder1Basis.cMat_i2
        self.cWeight_i2 = gid.GkeDgLobatto1DPolyOrder1Weights.cWeight_i2

    def project(self, c):
        qn = self._getRaw(c)
        return makeMesh(2, self.Xc[0]), interpOnMesh1D(self.cMat_i2, qn)

    def integrate(self, c):
        qn = self._getRaw(c)
        return computeIntegratedQuantity1D(self.cWeight_i2, qn, self.Xc[0])

#################
class GkeDgLobatto1DPolyOrder2Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 2 basis in 1D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 3)
        self.cMat_i3 = gid.GkeDgLobatto1DPolyOrder2Basis.cMat_i3
        self.cWeight_i3 = gid.GkeDgLobatto1DPolyOrder2Weights.cWeight_i3

    def project(self, c):
        qn = self._getRaw(c)
        return makeMesh(3, self.Xc[0]), interpOnMesh1D(self.cMat_i3, qn)

    def integrate(self, c):
        qn = self._getRaw(c)
        return computeIntegratedQuantity1D(self.cWeight_i3, qn, self.Xc[0])

#################
class GkeDgLobatto1DPolyOrder3Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 3 basis in 1D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 4)
        self.cMat_i4 = gid.GkeDgLobatto1DPolyOrder3Basis.cMat_i4
        self.cWeight_i4 = gid.GkeDgLobatto1DPolyOrder3Weights.cWeight_i4

    def project(self, c):
        qn = self._getRaw(c)
        return makeMesh(4, self.Xc[0]), interpOnMesh1D(self.cMat_i4, qn)

    def integrate(self, c):
        qn = self._getRaw(c)
        return computeIntegratedQuantity1D(self.cWeight_i4, qn, self.Xc[0])

#################
class GkeDgLobatto1DPolyOrder4Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 4 basis in 1D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 5)
        self.cMat_i5 = gid.GkeDgLobatto1DPolyOrder4Basis.cMat_i5
        self.cWeight_i5 = gid.GkeDgLobatto1DPolyOrder4Weights.cWeight_i5

    def project(self, c):
        qn = self._getRaw(c)
        return makeMesh(5, self.Xc[0]), interpOnMesh1D(self.cMat_i5, qn)

    def integrate(self, c):
        qn = self._getRaw(c)
        return computeIntegratedQuantity1D(self.cWeight_i5, qn, self.Xc[0])

#################
class GkeDgLobatto2DPolyOrder1Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 1 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 4)
        self.cMat_i2 = gid.GkeDgLobatto2DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(2, self.Xc[0]), makeMesh2(2, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i2, qn)

#################
class GkeDgLobatto2DPolyOrder2Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 2 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 9)
        self.cMat_i3 = gid.GkeDgLobatto2DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(3, self.Xc[0]), makeMesh2(3, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i3, qn)

#################
class GkeDgLobatto2DPolyOrder3Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 3 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 16)
        self.cMat_i4 = gid.GkeDgLobatto2DPolyOrder3Basis.cMat_i4

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(4, self.Xc[0]), makeMesh2(4, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i4, qn)

#################
class GkeDgLobatto2DPolyOrder4Basis(GkeDgBasis):
    r"""Lobatto, polyOrder = 4 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 5*5)
        self.cMat_i5 = gid.GkeDgLobatto2DPolyOrder4Basis.cMat_i5

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(5, self.Xc[0]), makeMesh2(5, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i5, qn)

#################
class GkeDgSerendip2DPolyOrder1Basis(GkeDgBasis):
    r"""Serendipity basis (Hakim layout), polyOrder = 1 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 4)
        self.cMat_i2 = gid.GkeDgSerendip2DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(2, self.Xc[0]), makeMesh2(2, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i2, qn)

#################
class GkeDgSerendip2DPolyOrder2Basis(GkeDgBasis):
    r"""Serendipity basis (Hakim layout), polyOrder = 2 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 8)
        self.cMat_i3 = gid.GkeDgSerendip2DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh2(3, self.Xc[0]), makeMesh2(3, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i3, qn)

#################
class GkeDgSerendipNorm2DPolyOrder1Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 1 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 4)
        self.cMat_i2 = gid.GkeDgSerendipNorm2DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh(2, self.Xc[0]), makeMesh(2, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i2, qn)

#################
class GkeDgSerendipNorm2DPolyOrder2Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 2 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 8)
        self.cMat_i3 = gid.GkeDgSerendipNorm2DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh(3, self.Xc[0]), makeMesh(3, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i3, qn)    

#################
class GkeDgSerendipNorm2DPolyOrder3Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 3 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 12)
        self.cMat_i4 = gid.GkeDgSerendipNorm2DPolyOrder3Basis.cMat_i4

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh(4, self.Xc[0]), makeMesh(4, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i4, qn) 

#################
class GkeDgSerendipNorm2DPolyOrder4Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 4 basis, in 2D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 17)
        self.cMat_i5 = gid.GkeDgSerendipNorm2DPolyOrder4Basis.cMat_i5

    def project(self, c):
        qn = self._getRaw(c)
        X, Y = makeMesh(5, self.Xc[0]), makeMesh(5, self.Xc[1])
        XX, YY = pylab.meshgrid(X, Y, indexing='ij')
        return XX, YY, interpOnMesh2D(self.cMat_i5, qn) 

#################
class GkeDgSerendipNorm3DPolyOrder1Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 1 basis, in 3D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 8)
        self.cMat_i2 = gid.GkeDgSerendipNorm3DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z = makeMesh(2, self.Xc[0]), makeMesh(2, self.Xc[1]), makeMesh(2, self.Xc[2])
        XX, YY, ZZ = pylab.meshgrid(X, Y, Z, indexing='ij')
        return XX, YY, ZZ, interpOnMesh3D(self.cMat_i2, qn)

#################
class GkeDgSerendipNorm3DPolyOrder2Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 2 basis, in 3D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 20)
        self.cMat_i3 = gid.GkeDgSerendipNorm3DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z = makeMesh(3, self.Xc[0]), makeMesh(3, self.Xc[1]), makeMesh(3, self.Xc[2])
        XX, YY, ZZ = pylab.meshgrid(X, Y, Z, indexing='ij')
        return XX, YY, ZZ, interpOnMesh3D(self.cMat_i3, qn)   

#################
class GkeDgSerendipNorm3DPolyOrder3Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 3 basis, in 3D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 32)
        self.cMat_i4 = gid.GkeDgSerendipNorm3DPolyOrder3Basis.cMat_i4

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z = makeMesh(4, self.Xc[0]), makeMesh(4, self.Xc[1]), makeMesh(4, self.Xc[2])
        XX, YY, ZZ = pylab.meshgrid(X, Y, Z, indexing='ij')
        return XX, YY, ZZ, interpOnMesh3D(self.cMat_i4, qn)

#################
class GkeDgSerendipNorm3DPolyOrder4Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 4 basis, in 3D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 50)
        self.cMat_i5 = gid.GkeDgSerendipNorm3DPolyOrder4Basis.cMat_i5

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z = makeMesh(5, self.Xc[0]), makeMesh(5, self.Xc[1]), makeMesh(5, self.Xc[2])
        XX, YY, ZZ = pylab.meshgrid(X, Y, Z, indexing='ij')
        return XX, YY, ZZ, interpOnMesh3D(self.cMat_i5, qn)

#################
class GkeDgSerendipNorm4DPolyOrder1Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 1 basis, in 4D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 16)
        self.cMat_i2 = gid.GkeDgSerendipNorm4DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z, V = makeMesh(2, self.Xc[0]), makeMesh(2, self.Xc[1]), makeMesh(2, self.Xc[2]), makeMesh(2, self.Xc[3])
        XX, YY, ZZ, VV = pylab.meshgrid(X, Y, Z, V, indexing='ij')
        return XX, YY, ZZ, VV, interpOnMesh4D(self.cMat_i2, qn)

#################
class GkeDgSerendipNorm4DPolyOrder2Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 2 basis, in 4D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 48)
        self.cMat_i3 = gid.GkeDgSerendipNorm4DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z, V = makeMesh(3, self.Xc[0]), makeMesh(3, self.Xc[1]), makeMesh(3, self.Xc[2]), makeMesh(3, self.Xc[3])
        XX, YY, ZZ, VV = pylab.meshgrid(X, Y, Z, V, indexing='ij')
        return XX, YY, ZZ, VV, interpOnMesh4D(self.cMat_i3, qn)

#################
class GkeDgSerendipNorm5DPolyOrder1Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 1 basis, in 5D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 32)
        self.cMat_i2 = gid.GkeDgSerendipNorm5DPolyOrder1Basis.cMat_i2

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z, V, U = makeMesh(2, self.Xc[0]), makeMesh(2, self.Xc[1]), makeMesh(2, self.Xc[2]), makeMesh(2, self.Xc[3]), makeMesh(2, self.Xc[4])
        XX, YY, ZZ, VV, UU = pylab.meshgrid(X, Y, Z, V, U, indexing='ij')
        return XX, YY, ZZ, VV, UU, interpOnMesh5D(self.cMat_i2, qn)

#################
class GkeDgSerendipNorm5DPolyOrder2Basis(GkeDgBasis):
    r"""Serendipity basis (correct, normal layout), polyOrder = 2 basis, in 5D
    """

    def __init__(self, dat):
        GkeDgBasis.__init__(self, dat, 112)
        self.cMat_i3 = gid.GkeDgSerendipNorm5DPolyOrder2Basis.cMat_i3

    def project(self, c):
        qn = self._getRaw(c)
        X, Y, Z, V, U = makeMesh(3, self.Xc[0]), makeMesh(3, self.Xc[1]), makeMesh(3, self.Xc[2]), makeMesh(3, self.Xc[3]), makeMesh(3, self.Xc[4])
        XX, YY, ZZ, VV, UU = pylab.meshgrid(X, Y, Z, V, U, indexing='ij')
        return XX, YY, ZZ, VV, UU, interpOnMesh5D(self.cMat_i3, qn)
