#!/usr/bin/env python
"""
postgkyl module containing history array classes CartFieldHist and
CartFieldDGHist
"""

# standart imports
import numpy
import exceptions
import os
import glob
# custom imports
import gkedgbasis
import plotting
import cartField

class CartFieldHist(object):
    """Base class for Gkeyll cartesian field time arrays loading and manipulation.

    Allows for batch manipulations.

    Methods:
    __init__ -- initialize class and load file if specified
    __del__  -- close opened HDF5 files
    close    -- close opened HDF5 files
    load     -- open Gkeyll output HDF5 file
    plot     -- plot the specified components of field
    """

    def __init__(self, fileNameBase=None):
        """Initialize the array of field and load HDF5 files if specified.
        
        Inputs:
        None
        
        Keyword arguments:
        fileNameBase -- HDF5 file name base (name without frame number)
                        to be opened (default None)
        """
        self.isLoaded = 0
        if fileNameBase is not None:
            self.load(fileNameBase)

    def __del__(self):
        """Close all HDF5 files if opened."""
        try:
            for field in self.history:
                field.close()
        except:
            pass

    def close(self):
        """Close all HDF5 files if opened."""
        try:
            for field in self.history:
                field.close()
        except:
            pass

    def load(self, fileNameBase):
        """Load Gkeyll HDF5 output files.

        Inputs:
        fileNameBase -- HDF5 file name base (name without frame number)
                        to be opened (default None)

        Keyword arguments:
        None

        Returns:
        None

        Note:
        Sets the isLoaded flag to 1.

        Exceptions:
        RuntimeError -- raise when specified file does not exist
        """
        self.fileNameBase = fileNameBase
        files = glob.glob('{}_*.h5'.format(self.fileNameBase))
        if files == []:
            raise exceptions.RuntimeError(
                'CartFieldHist:load Files with base \'{}\' does not exist!'.format(self.fileNameBase))
        self.isLoaded = 1
        # load history
        self.history = [cartField.CartField(name) for name in files]
        self.history = numpy.array(self.history)
        # sort history
        time = [temp.time for temp in self.history]
        idxSort = numpy.argsort(time)
        self.history = self.history[idxSort]

    def plot(self, numSnapshots, comp=0, 
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        """Plot time snapshots snapshots of the history data.
        
        Inputs:
        numSnapshots -- number of time snapshots to create
        
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
        plotting.plotFieldHist(self, numSnapshots, comp=comp,
                               fix1=fix1, fix2=fix2, fix3=fix3,
                               fix4=fix4, fix5=fix5, fix6=fix6)


class CartFieldDGHist(CartFieldHist):
    """Base class for Gkeyll cartesian DG field time arrays loading and manipulation.

    Parent: CartFieldHist

    Methods:
    __init__ -- initialize class and load files if specified
    load     -- open Gkeyll output HDF5 files
    plot     -- plot the specified components of fields
    projet   -- projet DG data based on basis and polynomial order
    """

    def __init__(self, basis, polyOrder, fileNameBase=None,
                 numComponents=1, loadProj=True):
        """Initialize the field array and load HDF5 files if specified.

        Inputs:
        basis     -- DG polynomial basis
        polyOrder -- polynomial order of DG approximation

        Keyword arguments:
        fileNameBase -- HDF5 file name base (name without frame number)
                        to be opened (default None)
        numComps     -- number of components of data (default 1)
        loadProj     -- flag to load projected data from file (default true)
        """
        self.isLoaded = 0
        self.basis     = basis
        self.polyOrder = polyOrder
        self.numComps  = numComponents
        self.isProj    = 0
        if fileNameBase is not None:
            self.load(fileNameBase, loadProj)

    def load(self, fileNameBase, loadProj=True):
        """Load Gkeyll HDF5 output files.

        Inputs:
        fileNameBase -- HDF5 file name base (name without frame number)
                        to be opened (default None)

        Keyword arguments:
        loadProj -- tries to load projected data from the file

        Returns:
        None

        Note:
        Sets the isLoaded flag to 1.

        Exceptions:
        RuntimeError -- raise when specified file does not exist
        """
        self.fileNameBase = fileNameBase
        files = glob.glob('{}_*.h5'.format(self.fileNameBase))
        if files == []:
            raise exceptions.RuntimeError(
                'CartFieldDGHist:load Files with base \'{}\' does not exist!'.format(self.fileNameBase))
        self.isLoaded = 1
        # load history
        self.history = [cartField.CartFieldDG(self.basis, self.polyOrder, 
                                              fileName=name, 
                                              numComponents=self.numComps,
                                              loadProj=loadProj) for name in files]
        self.history = numpy.array(self.history)
        # sort history
        time = [temp.time for temp in self.history]
        idxSort = numpy.argsort(time)
        self.history = self.history[idxSort]

    def project(self):
        """Project all data with an apropriate basis from the module GkeDgBasis

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
                "CartFieldDGHist.project: Data needs to be loaded first. Use CartField.load(fileName).")
        
        self.isProj = 1

        for field in self.history:
            field.project()

    def plot(self, numSnapshots, comp=0, noProj=None, 
             fix1=None, fix2=None, fix3=None,
             fix4=None, fix5=None, fix6=None):
        """Plot time snapshots snapshots of the history data.
        
        Inputs:
        numSnapshots -- number of time snapshots to create
        
        Keyword arguments:
        comp -- list or tuple of components to be plotted (default 0)
        noProj -- disable automated projection of data (default False)
        fix1 -- fixes the first coordinate to provided index (default None)
        fix2 -- fixes the second coordinate to provided index (default None)
        fix3 -- fixes the third coordinate to provided index (default None)
        fix4 -- fixes the fourth coordinate to provided index (default None)
        fix5 -- fixes the fifth coordinate to provided index (default None)
        fix6 -- fixes the sixth coordinate to provided index (default None)
        
        Returns:
        None
        
        Note: 
        If the projected data are not available, the project
        method will be called before plotting, unless specified
        otherwise by flag 'noProj'.
        """
        if noProj:
            plotting.plotFieldHist(self, numSnapshots, comp=comp,
                                   fix1=fix1, fix2=fix2, fix3=fix3,
                                   fix4=fix4, fix5=fix5, fix6=fix6)
        else:
            plotting.plotFieldHist(self, numSnapshots, comp=comp, isDG=True,
                                   fix1=fix1, fix2=fix2, fix3=fix3,
                                   fix4=fix4, fix5=fix5, fix6=fix6)
