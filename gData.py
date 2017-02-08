#!/usr/bin/env python
"""
Postgkyl module to load G* data
"""

import numpy
import os
import exceptions

class GData:
    """Provide interface to read output data.

    __init__(fName : string)
    Determine the data type and call the appropriate load function

    Methods:
    loadG1h5 -- Load G1 HDF5 file
    loadG2bp -- Load G2 Adios binary file
    """

    def __init__(self, fName):
        """Determine the data type and call the appropriate load
        function

        Inputs:
        fName -- file name

        Notes:
        Load function is determined based on the extension
        """
        self.fName = fName
        if not os.path.exists(self.fName):
            raise exceptions.RuntimeError(
                "GData: File {} does not exist!".format(fName))
        # Parse the file name and select the last part (extension)
        ext = self.fName.split('.')[-1]
        if ext == 'h5':
            self.loadG1h5()
        elif ext == 'bp':
            self.loadG2bp()
        else:
            raise exceptions.RuntimeError(
                "GData: File extension {} is not supported.".format(ext))

    def loadG1h5(self):
        """Load the G1 HDF5 file"""
        import tables

        fh   = tables.open_file(self.fName, 'r')
        grid = fh.root.StructGrid

        # read in information about grid
        self.lowerBounds = numpy.array(grid._v_attrs.vsLowerBounds)
        self.upperBounds = numpy.array(grid._v_attrs.vsUpperBounds)
        self.numCells    = numpy.array(grid._v_attrs.vsNumCells)
        self.numDims     = len(self.numCells)

        # read in time data if it exists
        try:
            self.time = numpy.float(fh.root.timeData._v_attrs.vsTime)
        except:
            self.time = numpy.float(0.0)
        
        # read in data
        self.q = numpy.array(fh.root.StructGridField)

        # close the opened file
        fh.close()

    def loadG2bp(self):
        """Load the G2 Adios binary file"""
        import adios

        fh = adios.file(self.fName)

        # read in information about grid
        # Note: when atribute is a scalar, ADIOS only returns standart
        # Python float; for consistency, those are turned into 1D
        # numpy arrays
        self.lowerBounds = numpy.array(adios.attr(fh, 'lowerBounds').value)
        if self.lowerBounds.ndim == 0:
            self.lowerBounds = numpy.expand_dims(self.lowerBounds, 0)
        self.upperBounds = numpy.array(adios.attr(fh, 'upperBounds').value)
        if self.upperBounds.ndim == 0:
            self.upperBounds = numpy.expand_dims(self.upperBounds, 0)
        self.numCells    = numpy.array(adios.attr(fh, 'numCells').value)
        if self.numCells.ndim == 0:
            self.numCells = numpy.expand_dims(self.numCells, 0)
        self.numDims     = len(self.numCells)

        # read in time data if it exists
        try:
            self.time = numpy.float(adios.readvar(self.fName, 'time'))
        except:
            self.time = numpy.float(0.0)
        
        # read in data
        self.q = adios.readvar(self.fName, 'CartGridField')
