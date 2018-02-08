import logging as log
from os.path import isfile
from glob import glob

import numpy as np
import tables
import adios


class GData(object):
    """Provides interface to Gkeyll output data.

    GData serves as a baseline interface to Gkeyll 1 and 2 output
    data. It allows for loading and saving and serves as an input for
    other Postgkyl methods. Represents a dataset in the Postgkyl
    command line mode.

    Attributes:

    """

    def __init__(self, fName, verbose=False):
        if verbose:
            log.basicConfig(format="%(levelname)s: %(message)s",
                            level=log.INFO)
        else:
            log.basicConfig(format="%(levelname)s: %(message)s")

        self._lower = []
        self._upper = []
        self._grid = []
        self._values = []

        self._fName = fName
        if isfile(self._fName):
            self._loadFrame()
        else:
            self._loadSequence()

    def _loadFrame(self):
        log.info(("Loading frame '{:s}'".format(self._fName)))

        extension = self._fName.split('.')[-1]
        if extension == 'h5':
            with tables.open_file(self._fName, 'r') as fh:
                try:
                    self._values.append(fh.root.StructGridField.read())
                except:
                    log.info(("'{:s}' is not a Gkeyll frame".
                              format(self._fName)))
                    fh.close()
                    self._loadSequence()
                    return
                lower = fh.root.StructGrid._v_attrs.vsLowerBounds
                upper = fh.root.StructGrid._v_attrs.vsUpperBounds
                cells = fh.root.StructGrid._v_attrs.vsNumCells
                try:
                    self.time = fh.root.timeData._v_attrs.vsTime
                except AttributeError:
                    self.time = None
        elif extension == 'bp':
            with adios.file(self._fName) as fh:
                try:
                    self._values.append(adios.readvar(self._fName,
                                                      'CartGridField'))
                except KeyError:
                    log.info(("'{:s}' is not a Gkeyll frame".
                              format(self._fName)))
                    self._loadSequence()
                    return
                lower = np.atleast_1d(
                    adios.attr(fh, 'lowerBounds').value)
                upper = np.atleast_1d(
                    adios.attr(fh, 'upperBounds').value)
                cells = np.atleast_1d(
                    adios.attr(fh, 'numCells').value)
                try:
                    self.time \
                        = adios.readvar(self._fName, 'time')
                except KeyError:
                    self.time = None

        else:
            raise NameError((
                "File extension '{:s}' is not supported".
                format(extension)))

        self._lower.append(lower)
        self._upper.append(upper)
        dr = (upper - lower) / cells
        grid = [np.linspace(lower[d] + 0.5*dr[d], upper[d] - 0.5*dr[d],
                            cells[d])
                for d in range(len(cells))]
        self._grid.append(grid)

    def _loadSequence(self):
        log.info(("Loading sequence '{:s}'".format(self._fName)))
        files = glob('{:s}*'.format(self._fName))
        if not files:
            raise NameError((
                "No data files with the root '{:s}'".
                format(self._fName)))

        numFilesLoaded = 0
        extension = files[0].split('.')[-1]
        if extension == 'h5':
            with tables.open_file(files[0], 'r') as fh:
                try:
                    self._values.append(fh.root.DataStruct.data.read())
                    self._grid.append(fh.root.DataStruct.timeMesh.read())
                    numFilesLoaded += 1
                except:
                    pass
        elif extension == 'bp':
            try:
                self._values.append(adios.readvar(files[0], 'Data'))
                self._grid.append(adios.readvar(files[0], 'TimeMesh'))
                numFilesLoaded += 1
            except KeyError:
                pass

        for fName in files[1:]:
            extension = fName.split('.')[-1]
            if extension == 'h5':
                with tables.open_file(fName, 'r') as fh:
                    try:
                        self._values[0] \
                            = np.append(self._values[0],
                                        fh.root.DataStruct.data.read(),
                                        axis=0)
                        self._grid[0] \
                            = np.append(self._grid[0],
                                        fh.root.DataStruct.timeMesh.read(),
                                        axis=0)
                        numFilesLoaded += 1
                    except:
                        pass
            elif extension == 'bp':
                try:
                    self._values[0] \
                        = np.append(self._values[0],
                                    adios.readvar(fName, 'Data'),
                                    axis=0)
                    self._grid[0] \
                        = np.append(self._grid[0],
                                    adios.readvar(fName, 'TimeMesh'),
                                    axis=0)
                    numFilesLoaded += 1
                except KeyError:
                    pass

        if numFilesLoaded == 0:
            raise NameError((
                "No data files with the root '{:s}'".
                format(self._fName)))

        if len(self._grid[0].shape) > 1:
            self._grid[0] = np.squeeze(self._grid[0])
        self._grid[0] = [self._grid[0]]  # following Postgkyl
                                         # convention and making
                                         # 'grid' a list of numpy
                                         # arrays for each dimension

        # glob() doesn't guarantee the right order
        sortIdx = np.argsort(self._grid[0][0])
        self._grid[0][0] = self._grid[0][0][sortIdx]
        self._values[0] = self._values[0][sortIdx, ...]

        # enforcing boundaries are arrays even for 1D data
        lower = np.expand_dims(np.array(self._grid[0][0][0]), 0)
        upper = np.expand_dims(np.array(self._grid[0][0][-1]), 0)
        self._lower.append(lower)
        self._upper.append(upper)
        self.time = None

    def peakGrid(self):
        return self._grid[-1], self._lower[-1], self._upper[-1]
    
    def popGrid(self):
        return self._grid.pop(), self._lower.pop(), self._upper.pop()
    
    def pushGrid(self, grid, lower=None, upper=None):
        self._grid.append(grid)
        if lower is not None:
            self._lower.append(lower)
        else:
            self._lower.append(self._lower[-1])
        if upper is not None:
            self._upper.append(upper)
        else:
            self._upper.append(self._upper[-1])

    def peakValues(self):
        return self._values[-1]
    
    def popValues(self):
        return self._grid.pop()
    
    def pushValues(self, values):
        self._values.append(values)
        
    def getNumDims(self):
        return len(self._grid[0])

    def getNumCells(self):
        numDims = self.getNumDims()
        cells = np.zeros(numDims, dtype=np.int)
        for d in range(numDims):
            cells[d] = len(self._grid[-1][d])
        return cells

    def getNumComps(self):
        return self._values[-1].shape[-1]

    def getBounds(self):
        return self._lower[-1], self._upper[-1]

    def info(self):
        """Prints GData object information.

        Prints time (only when available), number of components, dimension
        spans, extremes for a GData object. Only the top of the stack is
        printed.
        
        Args:
        
        Returns:
            output (str): A list of strings with the informations
        """
        values = self.peakValues()
        numComps = self.getNumComps()
        numDims = self.getNumDims()
        numCells = self.getNumCells()
        lower, upper = self.getBounds()

        maximum = values.max()
        maxIdx = np.unravel_index(np.argmax(values), values.shape)
        minimum = values.min()
        minIdx = np.unravel_index(np.argmin(values), values.shape)
        
        output = ""
        if self.time is not None:
            output += "- Time: {:e}\n".format(self.time)
        output += "- Number of components: {:d}\n".format(numComps)
        output += "- Number of dimensions: {:d}\n".format(numDims)
        for d in range(numDims):
            output += "  - Dim {:d}: Num. cells: {:d}; ".format(d, numCells[d])
            output += "Lower: {:e}; Upper: {:e}\n".format(lower[d], upper[d])
        output += "- Maximum: {:e} at {:s}".format(maximum,
                                                   str(maxIdx[:numDims]))
        if numComps > 1:
            output += " component {:d}\n".format(maxIdx[-1])
        else:
            output += "\n"
        output += "- Minimum: {:e} at {:s}".format(minimum,
                                                   str(minIdx[:numDims]))
        if numComps > 1:
            output += " component {:d}".format(minIdx[-1])
        return output
