#!/usr/bin/env python
"""
posgkyl module containing basic cartesian field classes CartField and
CartFieldDG
"""

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
            'serendipity nodal' : ((gkedgbasis.GkeDgSerendipNorm1DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm1DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipNorm1DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipNorm1DPolyOrder4Basis),
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
                              gkedgbasis.GkeDgSerendipNorm4DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipNorm4DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipNorm5DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipNorm5DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipNorm5DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipNorm5DPolyOrder4Basis)),
            'serendipity modal' : ((gkedgbasis.GkeDgSerendipModal1DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipModal1DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipModal1DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipModal1DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipModal2DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipModal2DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipModal2DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipModal2DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipModal3DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipModal3DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipModal3DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipModal3DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipModal4DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipModal4DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipModal4DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipModal4DPolyOrder4Basis),
                             (gkedgbasis.GkeDgSerendipModal5DPolyOrder1Basis,
                              gkedgbasis.GkeDgSerendipModal5DPolyOrder2Basis,
                              gkedgbasis.GkeDgSerendipModal5DPolyOrder3Basis,
                              gkedgbasis.GkeDgSerendipModal5DPolyOrder4Basis)),
            'maximal order modal' : ((gkedgbasis.GkeDgMaximalOrderModal1DPolyOrder1Basis,
                              gkedgbasis.GkeDgMaximalOrderModal1DPolyOrder2Basis,
                              gkedgbasis.GkeDgMaximalOrderModal1DPolyOrder3Basis,
                              gkedgbasis.GkeDgMaximalOrderModal1DPolyOrder4Basis),
                             (gkedgbasis.GkeDgMaximalOrderModal2DPolyOrder1Basis,
                              gkedgbasis.GkeDgMaximalOrderModal2DPolyOrder2Basis,
                              gkedgbasis.GkeDgMaximalOrderModal2DPolyOrder3Basis,
                              gkedgbasis.GkeDgMaximalOrderModal2DPolyOrder4Basis),
                             (gkedgbasis.GkeDgMaximalOrderModal3DPolyOrder1Basis,
                              gkedgbasis.GkeDgMaximalOrderModal3DPolyOrder2Basis,
                              gkedgbasis.GkeDgMaximalOrderModal3DPolyOrder3Basis,
                              gkedgbasis.GkeDgMaximalOrderModal3DPolyOrder4Basis),
                             (gkedgbasis.GkeDgMaximalOrderModal4DPolyOrder1Basis,
                              gkedgbasis.GkeDgMaximalOrderModal4DPolyOrder2Basis,
                              gkedgbasis.GkeDgMaximalOrderModal4DPolyOrder3Basis,
                              gkedgbasis.GkeDgMaximalOrderModal4DPolyOrder4Basis),
                             (gkedgbasis.GkeDgMaximalOrderModal5DPolyOrder1Basis,
                              gkedgbasis.GkeDgMaximalOrderModal5DPolyOrder2Basis,
                              gkedgbasis.GkeDgMaximalOrderModal5DPolyOrder3Basis,
                              gkedgbasis.GkeDgMaximalOrderModal5DPolyOrder4Basis))}
            
# Helper  methods
def fixCoordinates(coords, values,
                   fix1=None, fix2=None, fix3=None,
                   fix4=None, fix5=None, fix6=None):
    """Fix specified coordinates.

    Inputs:
    coords -- array of coordinates
    values -- array of field values

    Keyword arguments:
    fix1 -- fixes the first coordinate to provided index (default None)
    fix2 -- fixes the second coordinate to provided index (default None)
    fix3 -- fixes the third coordinate to provided index (default None)
    fix4 -- fixes the fourth coordinate to provided index (default None)
    fix5 -- fixes the fifth coordinate to provided index (default None)
    fix6 -- fixes the sixth coordinate to provided index (default None)

    Returns:
    coordsFix -- coordinates with decreased number of dimentsions
    valuesFix -- field values with decreased number of dimentsions

    Example:
    By fixing and x-index (fix1), 1X1V simulation data transforms
    to 1D.

    Note:
    Fixing higher dimensions than available in the data has no effect.
    """
    fix = (fix1, fix2, fix3, fix4, fix5, fix6)
    coordsFix = coords
    valuesFix = values
    for i, value in reversed(list(enumerate(fix))):
        if value is not None and len(values.shape) > i:
            mask = numpy.zeros(values.shape[i])
            mask[fix[i]] = 1
            # delete coordinate matrices for the fixed coordinate
            coordsFix = numpy.delete(coordsFix, i, 0)
            coordsFix = numpy.compress(mask, coordsFix, axis=i+1)  
            coordsFix = numpy.squeeze(coordsFix)

            valuesFix = numpy.compress(mask, valuesFix, axis=i) 
            valuesFix = numpy.squeeze(valuesFix)
    return coordsFix, valuesFix


class CartField(object):
    """Base class for Gkeyll cartesian fields loading and manipulation.

    Methods:
    __init__ -- initialize class and load file if specified
    __del__  -- close opened HDF5 file
    close    -- close opened HDF5 file
    load     -- open Gkeyll output HDF5 file
    plot     -- plot the specified components of field
    """

    def __init__(self, fileName=None):
        """Initialize the field and load HDF5 file if specified.

        Inputs:
        None

        Keyword arguments:
        fileName -- HDF5 file name to be opened (default None)
        """
        self.isLoaded = 0
        if fileName is not None:
            self.load(fileName)

    def __del__(self):
        """Close HDF5 file if opened."""
        try:
            self.fh.close()
        except:
            pass

    def close(self):
        """Close the HDF5 file"""
        self.fh.close()

    def load(self, fileName):
        """Load Gkeyll HDF5 output file.

        Inputs:
        fileName -- HDF5 file name to be opened

        Keyword arguments:
        None

        Returns:
        None

        Note:
        Sets the isLoaded flag to 1.

        Exceptions:
        RuntimeError -- raise when specified file does not exist
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

        # construct coordinate matrices
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
        """Plot the current field data.

        Calls the general field plotting function plotting.plotField

        Inputs:
        None

        Keyword arguments:
        comp -- list or tuple of components to be plotted (default 0)
        fix1 -- fixes the first coordinate to provided index (default None)
        fix2 -- fixes the second coordinate to provided index (default None)
        fix3 -- fixes the third coordinate to provided index (default None)
        fix4 -- fixes the fourth coordinate to provided index (default None)
        fix5 -- fixes the fifth coordinate to provided index (default None)
        fix6 -- fixes the sixth coordinate to provided index (default None)

        Returns:
        None
        """
        plotting.plotField(self, comp=comp,
                           fix1=fix1, fix2=fix2, fix3=fix3,
                           fix4=fix4, fix5=fix5, fix6=fix6)


class CartFieldDG(CartField):
    """Base class for Gkeyll cartesian DG fields loading and manipulation.

    Parent: CartField

    Methods:
    __init__ -- initialize class and load file if specified
    __del__  -- close opened HDF5 file
    load     -- open Gkeyll output HDF5 file
    plot     -- plot the specified components of field
    projet   -- projet DG data based on basis and polynomial order
    save     -- save the DG projected data to a HDF5 file
    """

    def __init__(self, basis, polyOrder, fileName=None, 
                 numComponents=1,loadProj=True):
        """Initialize the field and load HDF5 file if specified.

        Inputs:
        basis     -- DG polynomial basis
        polyOrder -- polynomial order of DG approximation

        Keyword arguments:
        fileName -- HDF5 file name to be opened (default None)
        numComps -- number of components of data (default 1)
        loadProj -- flag to load projected data from file (default true)
        """
        self.isLoaded = 0
        if fileName is not None:
            self.load(fileName, loadProj)
        self.basis     = basis
        self.polyOrder = polyOrder
        self.numComps  = numComponents
        self.isProj    = 0

    def load(self, fileName, loadProj=True):
        """Load Gkeyll HDF5 output file.

        Calls the parent's (CartField) load file.

        Inputs:
        fileName -- HDF5 file name to be opened

        Keyword arguments:
        loadProj -- flag to load projected data from file (default true)

        Returns:
        None

        Note:
        Sets the isLoaded flag to 1.

        Exceptions:
        RuntimeError -- raise when specified file does not exist
        """
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

    def plot(self, comp=0, noProj=False,
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        """Plot the current field data.

        Calls the general field plotting function plotting.plotField

        Inputs:
        None

        Keyword arguments:
        comp   -- list or tuple of components to be plotted (default 0)
        noProj -- disable automated projection of data (default False)
        fix1   -- fixes the first coordinate to provided index (default None)
        fix2   -- fixes the second coordinate to provided index (default None)
        fix3   -- fixes the third coordinate to provided index (default None)
        fix4   -- fixes the fourth coordinate to provided index (default None)
        fix5   -- fixes the fifth coordinate to provided index (default None)
        fix6   -- fixes the sixth coordinate to provided index (default None)

        Returns:
        None

        Note: 
        If the projected data are not available, the project
        method will be called before plotting, unless specified
        otherwise by flag 'noProj'.
        """
        if noProj:
            plotting.plotField(self, comp, isDG=False,
                               fix1=fix1, fix2=fix2, fix3=fix3,
                               fix4=fix4, fix5=fix5, fix6=fix6)
        else:
            plotting.plotField(self, comp, isDG=True,
                               fix1=fix1, fix2=fix2, fix3=fix3,
                               fix4=fix4, fix5=fix5, fix6=fix6)

    def project(self):
        """Project data with an apropriate basis from the module GkeDgBasis.

        Inputs:
        None

        Keywords:
        None

        Returns:
        None

        Note:
        Calculates projection for all the components specified
        by the 'numComps' variable during the initialization.
        """
        if not self.isLoaded:
            raise exceptions.RuntimeError(
                "CartFieldDG.project: Data needs to be loaded first. Use CartFieldDG.load(fileName).")
        
        self.isProj = 1

        if basisMap[self.basis][self.numDims-1][self.polyOrder-1] is not None:
            temp = basisMap[self.basis][self.numDims-1][self.polyOrder-1](self)
        else:
            raise ValueError(
                "CartFieldDG.project: Basis, dimension, polynomial order combination is not supported.")
        
        # project the zeroth component
        projection      = numpy.array(temp.project(0))
        self.coordsProj = numpy.squeeze(projection[0:self.numDims, :])
        self.qProj      = projection[-1, :]
        self.qProj      = numpy.expand_dims(self.qProj, axis=self.numDims)
        # project the potential additional components
        if self.numComps > 1:
            for i in numpy.arange(self.numComps-1)+1:
                projection = numpy.array(temp.project(i))
                projection = projection[-1, :]
                projection = numpy.expand_dims(projection, 
                                               axis=self.numDims)
                self.qProj = numpy.append(self.qProj, projection,
                                          axis=self.numDims)

    def save(self, saveAs=None):
        """Save projected data and coordinates into HDF5 file.

        This will be slightly modified soon.
        """
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

        


