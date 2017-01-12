#!/usr/bin/env python
# standart imports
import numpy
import exceptions
# custom imports
import gkedata
import gkedgbasis
import plotting

# projection map
basisMap = {'lobatto' : ((gkedgbasis.GkeDgLobatto1DPolyOrder1Basis,
                         gkedgbasis.GkeDgLobatto1DPolyOrder2Basis,
                         gkedgbasis.GkeDgLobatto1DPolyOrder3Basis,
                         gkedgbasis.GkeDgLobatto1DPolyOrder4Basis),
                        (gkedgbasis.GkeDgLobatto2DPolyOrder1Basis,
                         gkedgbasis.GkeDgLobatto2DPolyOrder2Basis,
                         gkedgbasis.GkeDgLobatto2DPolyOrder3Basis,
                         gkedgbasis.GkeDgLobatto2DPolyOrder4Basis)),
            'serendipity' : ((None, None, None, None),
                             (gkedgbasis.GkeDgSerendipNorm2DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm2DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipNorm2DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipNorm2DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipNorm3DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm3DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipNorm3DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipNorm3DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipNorm4DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm4DPolyOrder2Basis,
                              None,
                              None),
                             (gkedgbasis.GkeDgSerendipNorm5DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm5DPolyOrder2Basis,
                              None,
                              None))}

class CartField(object):
    r"""
    Base class for Gkeyll Cartesian fields
    """

    def __init__(self, fileName=None):
        if fileName is not None:
            self.load(fileName)

    def load(self, fileName):
        r"""
        Load Gkeyll output file using the gkedata class.
        """
        self.data = gkedata.GkeData(fileName)
        self.isLoaded = 1

    def plot(self, comp=0, cuts=None):
        r"""
        Plots the field data.
        """
        if not self.isLoaded:
            raise exceptions.RuntimeError(
                "CartField.plot: Data needs to be loaded first. Use CartField.load(fileName).")

        comp = numpy.array(comp)
        if comp.ndim == 0:
            comp = numpy.expand_dims(comp, 0)

        if cuts is None:
            if self.data.ndim == 1:
                x = numpy.linspace(self.data.lowerBounds[0], self.data.upperBounds[0], self.data.cells[0])
                for i in range(comp.size):
                    y = self.data.q[:, comp[i]]
                    plotting.plot1D(x, y)
                    
            elif self.data.ndim == 2:
                x = numpy.linspace(self.data.lowerBounds[0], self.data.upperBounds[0], self.data.cells[0])
                y = numpy.linspace(self.data.lowerBounds[1], self.data.upperBounds[1], self.data.cells[1])
                XX, YY = numpy.meshgrid(x, y)
                for i in range(comp.size):
                    ZZ = self.data.q[:, :, comp[i]]
                    plotting.plot2D(XX, YY, numpy.transpose(ZZ))
            else:
                raise exeptions.RuntimeError(
                    "CartField.plot: Dimension of the field is bigger than two - cuts need to be specified.")
        else:
            pass

class CartFieldDG(CartField):
    r"""
    Class for Gkeyll DG Cartesian fields
    """

    def __init__(self, basis, polyOrder, fileName=None):
        super(CartFieldDG, self).__init__(fileName)
        self.basis     = basis
        self.polyOrder = polyOrder
        self.isProj    = 0

    def project(self, component=0):
        r"""
        Project data with apropriate basis from GkeDgBasis
        """
        if not self.isLoaded:
            raise exceptions.RuntimeError(
                "CartField.project: Data needs to be loaded first. Use CartField.load(fileName).")
        
        self.isProj = 1

        if basisMap[self.basis][self.dim-1][self.polyOrder-1] is not None:
            temp = basisMap[self.basis][self.dim-1][self.polyOrder-1](self.data)
        else:
            raise ValueError(
                "CartField.project: Basis, dimension, polynomial order combination is not supported.")
        
        projection = numpy.array(temp.project(component))
        self.coords   = numpy.squeeze(projection[0:self.dims, :])
        self.dataProj = projection[-1, :]
        

        


