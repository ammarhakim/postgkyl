#!/usr/bin/env python
# standart imports
import numpy
import exceptions
import tables
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

    def __del__(self):
        if self.isLoaded:
            self.data.close()

    def load(self, fileName):
        r"""
        Load Gkeyll output file using the gkedata class.
        """
        self.isLoaded = 1
        self.fileName = fileName
        self.data = gkedata.GkeData(fileName)

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
        self.data.fh.close()

class CartFieldDG(CartField):
    r"""
    Class for Gkeyll DG Cartesian fields
    """

    def __init__(self, basis, polyOrder, fileName=None, loadProj=True):
        if fileName is not None:
            self.load(fileName, loadProj)
        self.basis     = basis
        self.polyOrder = polyOrder
        self.isProj    = 0

    def load(self, fileName, loadProj=True):
        super(CartFieldDG, self).load(fileName)
        if loadProj:
            try:
                self.dataProj = numpy.array(self.data.fh.root.StructGridFieldProj)
            except:
                pass

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
        if noProject:
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

        


