"""
postgkyl sub-module to load and save gkeyll/gkyl data
"""
import numpy
import os
from glob import glob


class GData:
    """Provide interface to read output data.

    Attributes:
        extension: A string indicating IO library (ADIOS/HDF5)
        fName: A string containing the file name
        lowerBounds:
        upperBounds:
        numDims:
        numCells:
        numComponents:
        q:
    """

    def __init__(self, fName, load=True):
        """Load data from a file

        Determines the data type (ADIOS/HDF5), loads header, and
        finally loads the data itself. The data load is optional. HDF5
        file is closed after the data load.

        Atributes:
            fName: file name

        Raises:
            NameError: when specified file is not found or the
                extension is not 'bp' or 'h5'
        """
        self.fName = fName
        if not os.path.exists(self.fName):
            raise NameError(
                "GData: File {:s} does not exist!".format(fName))

        self.extension = self.fName.split('.')[-1]
        if self.extension == 'h5':
            import tables

            fh = tables.open_file(self.fName, 'r')
            grid = fh.root.StructGrid

            self.lowerBounds = numpy.array(grid._v_attrs.vsLowerBounds)
            self.upperBounds = numpy.array(grid._v_attrs.vsUpperBounds)
            self.numCells = numpy.array(grid._v_attrs.vsNumCells)
            self.numDims = len(self.numCells)
            try:
                self.time = numpy.float(fh.root.timeData._v_attrs.vsTime)
            except:
                self.time = numpy.float(-1.0)
        elif self.extension == 'bp':
            import adios

            fh = adios.file(self.fName)

            # Note: When atribute is a scalar, ADIOS only returns standart
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
            try:
                self.time = numpy.float(adios.readvar(self.fName, 'time'))
            except:
                self.time = numpy.float(-1.0)
        else:
            raise NameError(
                "GData: File extension {:s} is not supported.".format(self.extension))

        if load:
            if self.extension == 'h5':  
                self.loadDataH5(fh)
                fh.close()
            elif self.extension == 'bp':
                self.loadDataBP()

    def loadDataH5(self, fh):
        """Load data from a HDF5 file"""

        self.q = numpy.array(fh.root.StructGridField)

        if len(self.q.shape) > self.numDims:
            self.numComponents = self.q.shape[-1]
        else:
            self.numComponents = 1


    def loadDataBP(self):
        """Load data from an ADIOS file"""

        self.q = adios.readvar(self.fName, 'CartGridField')

        if len(self.q.shape) > self.numDims:
            self.numComponents = self.q.shape[-1]
        else:
            self.numComponents = 1

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
        self.files = glob('{}*'.format(self.fNameRoot))
        for fl in self.files:
            ext = fl.split('.')[-1]
            if ext != 'h5' and ext != 'bp':
                self.files.remove(fl)
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
        self.values = numpy.array(fh.root.DataStruct.data.read())
        self.time = numpy.array(fh.root.DataStruct.timeMesh.read())
        self.time = numpy.squeeze(self.time)
        fh.close()
        # read the rest of the files and append
        for fl in self.files[start+1 :]:
            ext = fl.split('.')[-1]
            try:
                fh = tables.open_file(fl, 'r')
                self.values = numpy.append(self.values,
                                           fh.root.DataStruct.data.read(), axis=0)
                self.time = numpy.append(self.time,
                                         fh.root.DataStruct.timeMesh.read())
                fh.close()
            except:
                fh.close()

        # sort with scending time
        sortIdx = numpy.argsort(self.time)
        self.time = self.time[sortIdx]
        self.values = self.values[sortIdx]

        # convert to numpy arrays
        self.values = numpy.array(self.values)
        self.time = numpy.array(self.time)

    def _loadG2bp(self, start):
        """Load the G2 ADIOS history data file"""
        import adios

        # read the first history file                     
        self.values = adios.readvar(self.files[start], 'Data')
        self.time = adios.readvar(self.files[start], 'TimeMesh')
    
        # read the rest of the files and append
        for fl in self.files[start+1 :]:
            ext = fl.split('.')[-1]
            try:
                self.values = numpy.append(self.values,
                                           adios.readvar(fl, 'Data'),
                                           axis=0)
                self.time = numpy.append(self.time,
                                         adios.readvar(fl, 'TimeMesh'),
                                         axis=0)
            except:
                fh.close()

        # sort with scending time
        sortIdx = numpy.argsort(self.time)
        self.time = self.time[sortIdx]
        self.values = self.values[sortIdx]

        # convert to numpy arrays
        self.values = numpy.array(self.values)
        self.time = numpy.array(self.time)

    def save(self, fName=None):
        """Write loaded history data to one text file

        Parameters:
        fName (optional) -- specify the output file name

        Note:
        If the 'fName' is not specified, 'fNameRoot' is used instead
        to construct a file name
        """
        if fName is None:
            fName = '{:s}/{:s}.dat'.format(os.getcwd(), self.fNameRoot)

        out = numpy.vstack([self.time, self.values]).transpose()
        numpy.savetxt(fName, out)
