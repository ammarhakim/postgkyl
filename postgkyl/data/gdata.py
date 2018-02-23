from glob import glob
from os.path import isfile
import logging as log

import adios
import numpy as np
import tables


class GData(object):
    """Provides interface to Gkeyll output data.

    GData serves as a baseline interface to Gkeyll 1 and 2 output
    data. It allows for loading and saving and serves as an input for
    other Postgkyl methods. Represents a dataset in the Postgkyl
    command line mode.
    """

    def __init__(self, fName, stack=False, comp=None,
                 coord0=None, coord1=None, coord2=None,
                 coord3=None, coord4=None, coord5=None):
        self._stack = stack  # Turn OFF the stack
        self._lower = []  # grid lower edges
        self._upper = []  # grid upper edges
        self._grid = []  # list of 1D grid slices
        self._values = []  # (N+1)D narray of values 

        self.fName = fName
        # Sequence load typically cancatenates multiple files
        # When the sequence is in just a single file, _loadFrame will
        # fail and _loadSequence is called instead
        if isfile(self.fName):
            coords = (coord0, coord1, coord2, coord3, coord4, coord5)
            self._loadFrame(coords, comp)
        else:
            self._loadSequence()

    #-----------------------------------------------------------------
    #-- File Loading -------------------------------------------------
    def _createOffsetCountBp(self, var, coords, comp):
        numDims = len(var.dims)
        count = np.array(var.dims)
        offset = np.zeros(numDims, np.int)

        for d, coord in enumerate(coords):
            if d < numDims-1 and coord is not None:  # Last dim stores comp
                if isinstance(coord, int):
                    offset[d] = coord
                    count[d] = 1
                elif isinstance(coord, str):
                    if len(coord.split(':')) == 2:
                        idxs = coord.split(':')
                        offset[d] = int(idxs[0])
                        count[d] = int(idxs[1]) - int(idxs[0])
                    else:
                        offset[d] = int(coord)
                        count[d] = 1
                else:
                    raise TypeError("'coord' is neither 'int' or 'str'")
        if comp is not None:
            if isinstance(comp, int):
                offset[-1] = comp
                count[-1] = 1
            elif isinstance(comp, str):
                if len(comp.split(':')) == 2:
                    idxs = comp.split(':')
                    offset[-1] = int(idxs[0])
                    count[-1] = int(idxs[1]) - int(idxs[0])
                else:
                    offset[-1] = int(comp)
                    count[-1] = 1
            else:
                raise TypeError("'comp' is neither 'int' or 'str'")
        return tuple(offset), tuple(count)

    def _loadFrame(self, axes=(None, None, None, None, None, None),
                   comp=None):
        extension = self.fName.split('.')[-1]
        # Gkeyll HDF5 file load
        if extension == 'h5':
            # 'with' is the prefered Python way to open a file which
            # can fail
            with tables.open_file(self.fName, 'r') as fh:
                try:
                    self._values.append(fh.root.StructGridField.read())
                except:
                    fh.close()
                    # Not a Gkyl "frame" data; trying to load as a sequence
                    self._loadSequence()
                    return

                lower = fh.root.StructGrid._v_attrs.vsLowerBounds
                upper = fh.root.StructGrid._v_attrs.vsUpperBounds
                cells = fh.root.StructGrid._v_attrs.vsNumCells
                try:
                    self.time = fh.root.timeData._v_attrs.vsTime
                except AttributeError:
                    self.time = None
        # Gkyl ADIOS file load
        elif extension == 'bp':
            # 'with' is the prefered Python way to open a file which
            # can fail
            with adios.file(self.fName) as fh:
                try:
                    var = adios.var(fh, 'CartGridField')
                except AssertionError:
                    # Not a Gkyl "frame" data; trying to load as a sequence
                    self._loadSequence()  
                    return
                # Get the atributes
                # Postgkyl conventions require the atribuest to be
                # narrays even for 1D data
                lower = np.atleast_1d(
                    adios.attr(fh, 'lowerBounds').value)
                upper = np.atleast_1d(
                    adios.attr(fh, 'upperBounds').value)
                cells = np.atleast_1d(
                    adios.attr(fh, 'numCells').value)
                # Create 'offset' and 'count' tuples
                offset, count = self._createOffsetCountBp(var, axes, comp)
                self._values.append(var.read(offset=offset, count=count))
                # Try loading the time-stamp
                try:
                    self.time \
                        = adios.readvar(self.fName, 'time')
                except KeyError:
                    self.time = None

            # Adjust boundaries for 'offset' and 'count'
            numDims = len(cells)
            dz = (upper - lower) / cells
            if offset:
                for d in range(numDims):
                    lower[d] = lower[d] + offset[d]*dz[d]
                    cells[d] = cells[d] - np.int(offset[d])
            if count:
                for d in range(numDims):
                    upper[d] = lower[d] + count[d]*dz[d]
                    cells[d] = np.int(count[d])
                            
        else:
            raise NameError((
                "File extension '{:s}' is not supported".
                format(extension)))

        numDims = len(cells)
        dz = (upper - lower) / cells
        # Adjusts bounds in case ghost layer is included in data
        for d in range(numDims):
            if cells[d] != self._values[0].shape[d]:
                ngl = int(np.floor((cells[d] - self._values[0].shape[d])*0.5))
                ngu = int(np.ceil((cells[d] - self._values[0].shape[d])*0.5))
                cells[d] = self._values[0].shape[d]
                lower[d] = lower[d] - ngl*dz[d]
                upper[d] = upper[d] + ngu*dz[d]

        self._lower.append(lower)
        self._upper.append(upper)

        # Create and append grid
        dz = (upper - lower) / cells
        grid = [np.linspace(lower[d] + 0.5*dz[d], upper[d] - 0.5*dz[d],
                            cells[d])
                for d in range(numDims)]
        self._grid.append(grid)

    def _loadSequence(self):
        # Sequence load typically cancatenates multiple files
        files = glob('{:s}*'.format(self.fName))
        if not files:
            raise NameError((
                "No data files with the root '{:s}'".
                format(self.fName)))

        cnt = 0  # Counter for the number of loaded files
        # Load the first file
        extension = files[0].split('.')[-1]
        # Gkeyll HDF5 load
        if extension == 'h5':
            with tables.open_file(files[0], 'r') as fh:
                try:
                    self._values.append(fh.root.DataStruct.data.read())
                    self._grid.append(fh.root.DataStruct.timeMesh.read())
                    cnt += 1
                except:
                    pass
        # Gkyl ADIOS load
        elif extension == 'bp':
            try:
                self._values.append(adios.readvar(files[0], 'Data'))
                self._grid.append(adios.readvar(files[0], 'TimeMesh'))
                cnt += 1
            except KeyError:
                pass
        
        # Load the rest of the files
        for fName in files[1:]:
            extension = fName.split('.')[-1]
            # Gkeyll HDF5 load
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
                        cnt += 1
                    except:  # PNG or TXT file can be offten globed as well
                        pass
            # Gkyl ADIOS load
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
                    cnt += 1
                except KeyError:  # PNG or TXT file can be offten globed
                    pass

        if cnt == 0:  # No files loaded
            raise NameError((
                "No data files with the root '{:s}'".
                format(self.fName)))

        # Squeeze the time coordinate ...
        if len(self._grid[0].shape) > 1:
            self._grid[0] = np.squeeze(self._grid[0])
        # ... and make it a list following the Postgkyl grid conventions
        self._grid[0] = [self._grid[0]]

        # glob() doesn't guarantee the right order
        sortIdx = np.argsort(self._grid[0][0])
        self._grid[0][0] = self._grid[0][0][sortIdx]
        self._values[0] = self._values[0][sortIdx, ...]

        # Boundaries should arrays even for 1D data
        lower = np.atleast_1d(self._grid[0][0][0])
        upper = np.atleast_1d(self._grid[0][0][-1])
        self._lower.append(lower)
        self._upper.append(upper)
        self.time = None  # Sequence has no time-stamp

    #-----------------------------------------------------------------
    #-- Stack Control ------------------------------------------------
    def getBounds(self):
        return self._lower[-1], self._upper[-1]

    def getNumCells(self):
        numDims = self.getNumDims()
        cells = np.zeros(numDims, dtype=np.int)
        for d in range(numDims):
            cells[d] = len(self._grid[-1][d])
        return cells

    def getNumComps(self):
        return self._values[-1].shape[-1]

    def getNumDims(self):
        return len(self._grid[0])

    def peakGrid(self):
        return self._grid[-1]

    def peakValues(self):
        return self._values[-1]

    def popGrid(self):
        if not self._stack:
            return self._grid.pop()
        else:
            raise RuntimeError("'pop' is dissables when stack is turned OFF")

    def popValues(self):
        if not self._stack:
            return self._grid.pop()
        else:
            raise RuntimeError("'pop' is dissables when stack is turned OFF")

    def pushGrid(self, grid, lower=None, upper=None):
        if not self._stack:
            self._grid.append(grid)
            if lower is not None:
                self._lower.append(lower)
            else:
                self._lower.append(self._lower[-1])
            if upper is not None:
                self._upper.append(upper)
            else:
                self._upper.append(self._upper[-1])
        else:
            self._grid[0] = grid
            if lower is not None:
                self._lower[0] = lower
            if upper is not None:
                self._upper[0] = upper

    def pushValues(self, values):
        if not self._stack:
            self._values.append(values)
        else:
            self._values[0] = values

    #-----------------------------------------------------------------
    #-- Output -------------------------------------------------------
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
            output += "  - Dim {:d}: Num. cells: {:d}; ".format(d,
                                                                numCells[d])
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
