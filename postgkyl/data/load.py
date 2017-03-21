#!/usr/bin/env python
"""
Postgkyl sub-module to load and save G* data
"""
import numpy
import os
import glob

class GData:
    """Provide interface to read output data.

    __init__(fName : string)
    Determine the data type and call the appropriate load function

    Methods:
    _loadG1h5 -- Load G1 HDF5 file
    _loadG2bp -- Load G2 Adios binary file
    """

    def __init__(self, fName):
        """Determine the data type and call the appropriate load
        function

        Parameters:
        fName -- file name

        Raises:
        NameError -- when specified file is not found
        NameError -- when file extension is neither h5 or bp

        Notes:
        Load function is determined based on the extension
        """
        self.fName = fName
        if not os.path.exists(self.fName):
            raise NameError(
                "GData: File {} does not exist!".format(fName))
        # Parse the file name and select the last part (extension)
        ext = self.fName.split('.')[-1]
        if ext == 'h5':
            self._loadG1h5()
        elif ext == 'bp':
            self._loadG2bp()
        else:
            raise NameError(
                "GData: File extension {} is not supported.".format(ext))

    def _loadG1h5(self):
        """Load the G1 HDF5 file"""
        import tables

        fh = tables.open_file(self.fName, 'r')
        grid = fh.root.StructGrid

        # read in information about grid
        self.lowerBounds = numpy.array(grid._v_attrs.vsLowerBounds)
        self.upperBounds = numpy.array(grid._v_attrs.vsUpperBounds)
        self.numCells = numpy.array(grid._v_attrs.vsNumCells)
        self.numDims  = len(self.numCells)

        # read in time data if it exists
        try:
            self.time = numpy.float(fh.root.timeData._v_attrs.vsTime)
        except:
            self.time = numpy.float(0.0)
        
        # read in data
        self.q = numpy.array(fh.root.StructGridField)

        # close the opened file
        fh.close()

    def _loadG2bp(self):
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
        self.numCells = numpy.array(adios.attr(fh, 'numCells').value)
        if self.numCells.ndim == 0:
            self.numCells = numpy.expand_dims(self.numCells, 0)
        self.numDims = len(self.numCells)

        # read in time data if it exists
        try:
            self.time = numpy.float(adios.readvar(self.fName, 'time'))
        except:
            self.time = numpy.float(0.0)
        
        # read in data
        self.q = adios.readvar(self.fName, 'CartGridField')


class GHistoryData:
    """Provide interface to read history data.

    __init__(fNameRoot : string)
    Determine the data type and call the appropriate load function

    Methods:
    _loadG1h5 -- Load G1 HDF5 files
    _loadG2bp -- Load G2 Adios binary files
    save      -- Save loaded data to a text file
    """

    def __init__(self, fNameRoot, start=0):
        """Determine the data type and call the appropriate load
        function

        Inputs:
        fNameRoot -- file name root

        Raises:
        NameError -- when files with root don't exis
        NameError -- when file extension is neither h5 or bp

        Notes:
        Load function is determined based on the extension
        """
        self.fNameRoot = fNameRoot
        self.files = glob.glob('*{}*.??'.format(self.fNameRoot))
        if self.files == []:
            raise NameError(
                'GHistoryData: Files with root \'{}\' do not exist!'.
                  format(self.fNameRoot))

        # Parse the file name and select the last part (extension)
        ext = self.files[start].split('.')[-1]
        if ext == 'h5':
            self._loadG1h5(start)
        elif ext == 'bp':
            self._loadG2bp(start)
        else:
            raise NameError(
                "GData: File extension {} is not supported.".format(ext))

    def _loadG1h5(self, start):
        """Load the G1 HDF5 history data file"""
        import tables

        # read the first history file
        fh = tables.open_file(self.files[start], 'r')
        self.values = fh.root.DataStruct.data.read()
        self.time = fh.root.DataStruct.timeMesh.read()
        fh.close()
        # read the rest of the files and append
        for file in self.files[start+1 :]:
            fh = tables.open_file(file, 'r')
            self.values = numpy.append(self.values,
                                       fh.root.DataStruct.data.read())
            self.time = numpy.append(self.time,
                                     fh.root.DataStruct.timeMesh.read())
            fh.close()

        # sort with scending time
        sortIdx = numpy.argsort(self.time)
        self.time = self.time[sortIdx]
        self.values = self.values[sortIdx]

        # convert to numpy arrays
        self.values = numpy.array(self.values)
        self.time = numpy.array(self.time)

    def _loadG2bp(self, start):
        print('ADIOS history data not yet suported')

    def save(self, fName=None):
        """Write loaded history data to one text file

        Parameters:
        fName (optional) -- specify the output file name

        Note:
        If the 'fName' is not specified, 'fNameRoot' is used instead
        to construct a file name
        """
        if fName is None:
            fName = '{}/{}.dat'.format(os.getcwd(), self.fNameRoot)

        out = numpy.vstack([self.time, self.values]).transpose()
        numpy.savetxt(fName, out)
