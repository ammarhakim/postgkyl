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

# helper functions
def fixCoordinates(coords, values,
                   fix1=None, fix2=None, fix3=None,
                   fix4=None, fix5=None, fix6=None):
    """
    Fixes specified coordinates and decreases the dimension of data.
    """
    fix = (fix1, fix2, fix3, fix4, fix5, fix6)
    coordsFix = coords
    valuesFix = values
    for i, value in reversed(list(enumerate(fix))):
        if value is not None and len(values.shape) > i:
            mask = numpy.zeros(values.shape[i])
            mask[fix[i]] = 1

            coordsFix = numpy.delete(coordsFix, i, 0)
            coordsFix = numpy.compress(mask, coordsFix, axis=i+1)  
            coordsFix = numpy.squeeze(coordsFix)

            valuesFix = numpy.compress(mask, valuesFix, axis=i) 
            valuesFix = numpy.squeeze(valuesFix)
    return coordsFix, valuesFix
                      
class CartField(object):
    r"""
    Base class for Gkeyll Cartesian fields
    """

    def __init__(self, fileName=None):
        if fileName is not None:
            self.load(fileName)

    def __del__(self):
        if self.isLoaded:
            self.data.close()

    def load(self, fileName):
        r"""
        Load Gkeyll output file using the gkedata class.
        """
        self.data = gkedata.GkeData(fileName)
        self.isLoaded = 1

    def plot(self, comp=0, 
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        r"""
        Plots the field data.
        """
        if not self.isLoaded:
            raise exceptions.RuntimeError(
                "CartField.plot: Data needs to be loaded first. Use CartField.load(fileName).")

        coords = []
        for i in range(self.data.ndim):
            temp = numpy.linspace(self.data.lowerBounds[i], self.data.upperBounds[i], self.data.cells[i])
            coords.append(temp)
        coordsMat = numpy.array(numpy.meshgrid(*coords, indexing='ij'))
        
        comp = numpy.array(comp)
        if comp.ndim == 0:
            comp = numpy.expand_dims(comp, 0)
        for i in range(comp.size):
            mask    = numpy.zeros(self.data.q.shape[self.data.ndim])
            mask[i] = 1
            values = numpy.compress(mask, self.data.q, self.data.ndim)
            values = numpy.squeeze(values)
            coordsPlot, valuesPlot = fixCoordinates(coordsMat, values,
                                                    fix1, fix2, fix3, fix4, fix5, fix6)
            if len(valuesPlot.shape) == 1:
                plotting.plot1D(coordsPlot, valuesPlot)
            elif len(valuesPlot.shape) == 2:
                plotting.plot2D(numpy.transpose(coordsPlot[0]), numpy.transpose(coordsPlot[1]),
                                numpy.transpose(valuesPlot))
            else:
                raise exeptions.RuntimeError(
                    "CartField.plot: Dimension of the field is bigger than two. Some dimensions need to be fixed.")

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

        if basisMap[self.basis][self.data.ndim-1][self.polyOrder-1] is not None:
            temp = basisMap[self.basis][self.data.ndim-1][self.polyOrder-1](self.data)
        else:
            raise ValueError(
                "CartField.project: Basis, dimension, polynomial order combination is not supported.")
        
        projection = numpy.array(temp.project(component))
        self.coords   = numpy.squeeze(projection[0:self.data.ndim, :])
        self.dataProj = projection[-1, :]

    def plot(self, comp=0, noProject=None,
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        r"""
        Plots the DG field data.
        """
        if noProject is not None:
            super(CartFieldDG, self).plot(comp, fix1, fix2, fix3, fix4, fix5, fix6)
            return

        if not self.isProj:
            print("Data not projected, projecting.")
            self.project()
        
        coordsPlot, valuesPlot = fixCoordinates(self.coords, self.dataProj,
                                                fix1, fix2, fix3, fix4, fix5, fix6)
        if len(valuesPlot.shape) == 1:
            plotting.plot1D(coordsPlot, valuesPlot)
        elif len(valuesPlot.shape) == 2:
            plotting.plot2D(numpy.transpose(coordsPlot[0]), numpy.transpose(coordsPlot[1]),
                            numpy.transpose(valuesPlot))
        else:
            raise exeptions.RuntimeError(
                "CartField.plot: Dimension of the field is bigger than two. Some dimensions need to be fixed.")


