#!/usr/bin/env python
# standart imports
import numpy
import exceptions
import tables
import os
# custom imports
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

# Helper  methods
def fixCoordinates(coords, values,
                   fix1=None, fix2=None, fix3=None,
                   fix4=None, fix5=None, fix6=None):
    r"""
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
        self.isLoaded = 0
        if fileName is not None:
            self.load(fileName)

    def __del__(self):
        try:
            self.fh.close()
        except:
            pass

    def load(self, fileName):
        r"""
        Load Gkeyll output file specified by 'fileName'.
        """
        self.fileName = fileName
        if not os.path.exists(self.fileName):
            raise exceptions.RuntimeError(
                'CartField:load File \'{}\' does not exist!'.format(fileName))
        self.isLoaded = 1
        self.fh = tables.open_file(fileName, 'a')
        grid = self.fh.root.StructGrid

        # read in information about grid
        self.lowerBounds = grid._v_attrs.vsLowerBounds
        self.upperBounds = grid._v_attrs.vsUpperBounds
        self.numCells    = grid._v_attrs.vsNumCells
        self.numDims     = len(self.numCells)

        # construct coordinates
        coords = []
        for i in range(self.numDims):
            temp = numpy.linspace(self.lowerBounds[i], 
                                  self.upperBounds[i],
                                  self.numCells[i])
            coords.append(temp)
        self.coords = numpy.array(numpy.meshgrid(*coords, indexing='ij'))

        # read in time data if it exists
        try:
            self.time = self.fh.root.timeData._v_attrs.vsTime
        except:
            self.time = 0.0
        
        # read in data
        self.q = self.fh.root.StructGridField

    def plot(self, comp=0, 
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        r"""
        Plots the field data.
        """
        plotting.plotField(self, comp,
                           fix1, fix2, fix3, fix4, fix5, fix6)

    def close(self):
        r"""
        Closes the HDF5 file
        """
        self.fh.close()

class CartFieldDG(CartField):
    r"""
    Class for Gkeyll DG Cartesian fields
    """

    def __init__(self, basis, polyOrder, fileName=None, 
                 numComponents=1,loadProj=True):
        self.isLoaded = 0
        if fileName is not None:
            self.load(fileName, loadProj)
        self.basis     = basis
        self.polyOrder = polyOrder
        self.numComps  = numComponents
        self.isProj    = 0

    def load(self, fileName, loadProj=True):
        super(CartFieldDG, self).load(fileName)
        if loadProj:
            try:
                self.qProj = \
                    numpy.array(self.fh.root.StructGridFieldProj)
                self.coordsProj = \
                    numpy.array(self.fh.root.StructGridFieldCoordsProj)
                self.isProj = 1
            except:
                pass

    def project(self):
        r"""
        Project data with apropriate basis from GkeDgBasis
        """
        if not self.isLoaded:
            raise exceptions.RuntimeError(
                "CartField.project: Data needs to be loaded first. Use CartField.load(fileName).")
        
        self.isProj = 1

        if basisMap[self.basis][self.numDims-1][self.polyOrder-1] is not None:
            temp = basisMap[self.basis][self.numDims-1][self.polyOrder-1](self)
        else:
            raise ValueError(
                "CartField.project: Basis, dimension, polynomial ",
                "order combination is not supported.")
        
        projection      = numpy.array(temp.project(0))
        self.coordsProj = numpy.squeeze(projection[0:self.numDims, :])
        self.qProj      = projection[-1, :]
        self.qProj      = numpy.expand_dims(self.qProj, axis=self.numDims)

        if self.numComps > 1:
            for i in numpy.arange(self.numComps-1)+1:
                projection = numpy.array(temp.project(i))
                projection = projection[-1, :]
                projection = numpy.expand_dims(projection, 
                                               axis=self.numDims)
                self.qProj = numpy.append(self.qProj, projection,
                                          axis=self.numDims)

    def plot(self, comp=0, noProj=None,
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        r"""
        Plots the DG field data.
        """
        if noProj:
            plotting.plotField(self, comp,
                               fix1, fix2, fix3, fix4, fix5, fix6)
        else:
            plotting.plotField(self, comp, True,
                               fix1, fix2, fix3, fix4, fix5, fix6)


    def save(self, saveAs=None):
        if saveAs is None:
            try:
                self.data.fh.remove_node("/StructGridFieldProj")
            except:
                pass
            self.data.fh.create_array("/", 'StructGridFieldProj', self.dataProj, 'Projected DG data field')
            self.data.fh.flush()
        else:
            self.data.fh.copy_file(saveAs, overwrite=True)
            fh = tables.open_file(saveAs, 'a')
            fh.create_array("/", 'StructGridFieldProj', self.dataProj, 'Projected DG data field')
            fh.flush()
            fh.close()

        


