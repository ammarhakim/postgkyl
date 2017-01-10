import matplotlib.pyplot as plt
import numpy

import gkedata
import gkedgbasis

class Field:
    r"""
    Base class for Gkeyll fields
    """

    def __init__(self, dim, basis, polyOrder, fileName=None):
        self.dim       = dim
        self.basis     = basis
        self.polyOrder = polyOrder
        self.isProj    = 0
        if fileName is not None:
            self.data = gkedata.GkeData(fileName)

    def load(self, fileName):
        r"""
        Load Gkeyll output file using the gkedata class.
        """
        self.data = gkedata.GkeData(fileName)

    def project(self, component=0):
        r"""
        Project data with apropriate basis from GkeDgBasis
        """
        self.isProj = 1

        if self.basis == 'lobatto':
            if self.dim == 1:
                if self.polyOrder == 1:
                    temp = gkedgbasis.GkeDgLobatto1DPolyOrder1Basis(self.data)
                    self.coord1, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 2:
                    temp = gkedgbasis.GkeDgLobatto1DPolyOrder2Basis(self.data)
                    self.coord1, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 3:
                    temp = gkedgbasis.GkeDgLobatto1DPolyOrder3Basis(self.data)
                    self.coord1, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 4:
                    temp = gkedgbasis.GkeDgLobatto1DPolyOrder4Basis(self.data)
                    self.coord1, self.dataProj = \
                        temp.project(component)
                else:
                    raise ValueError("Basis, dimension, polynomial order",
                                     "is not supported.")
            elif self.dim == 2:
                if self.polyOrder == 1:
                    temp = gkedgbasis.GkeDgLobatto2DPolyOrder1Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 2:
                    temp = gkedgbasis.GkeDgLobatto2DPolyOrder2Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 3:
                    temp = gkedgbasis.GkeDgLobatto2DPolyOrder3Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 4:
                    temp = gkedgbasis.GkeDgLobatto2DPolyOrder4Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                else:
                    raise ValueError("Basis, dimension, polynomial order",
                                     "is not supported.")
            else:
                raise ValueError("Basis, dimension, polynomial order",
                                 "is not supported.")
        elif self.basis == 'serendipity':
            if self.dim == 1:
                raise ValueError("Basis, dimension, polynomial order",
                                 "is not supported.")
            elif self.dim == 2:
                if self.polyOrder == 1:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm2DPolyOrder1Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 2:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm2DPolyOrder2Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 3:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm2DPolyOrder3Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                elif self.polyOrder == 4:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm2DPolyOrder4Basis(self.data)
                    self.coord1, self.coord2, self.dataProj = \
                        temp.project(component)
                else:
                    raise ValueError("Basis, dimension, polynomial order",
                                     "is not supported.")
            elif self.dim == 4:
                if self.polyOrder == 1:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm4DPolyOrder1Basis(self.data)
                    self.coord1, self.coord2, self.coord3, self.coord4, \
                        self.dataProj = temp.project(component)
                elif self.polyOrder == 2:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm4DPolyOrder2Basis(self.data)
                    self.coord1, self.coord2, self.coord3, self.coord4, \
                        self.dataProj = temp.project(component)
                else:
                    raise ValueError("Basis, dimension, polynomial order",
                                     "is not supported.")
            elif self.dim == 5:
                if self.polyOrder == 1:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm5DPolyOrder1Basis(self.data)
                    self.coord1, self.coord2, self.coord3, self.coord4, \
                        self.coord5, self.dataProj = temp.project(component)
                elif self.polyOrder == 2:
                    temp = \
                        gkedgbasis.GkeDgSerendipNorm5DPolyOrder2Basis(self.data)
                    self.coord1, self.coord2, self.coord3, self.coord4, \
                        self.coor5, self.dataProj = temp.project(component)
                else:
                    raise ValueError("Basis, dimension, polynomial order",
                                     "is not supported.")
            else:
                raise ValueError("Basis, dimension, polynomial order",
                                 "is not supported.")
        else:
            raise ValueError("Basis, dimension, polynomial order",
                             "is not supported.")
