#from difflib import SequenceMatcher
from glob import glob
from os.path import isfile
import adios
import numpy as np
import shutil
import sys
import tables

from postgkyl.utils import idxParser

# gets grid file name given field name and grid name
# def getGridFileName(fName, gridName):
#     fl = glob("*%s.bp*" % gridName)
#     for f in fl:
#         m = SequenceMatcher(None, fName, fName, f).find_longest_match(0, len(fName), 0, len(f))
#         if m.a == 0 and m.b == 0 and m.size > 0:
#             return f
#     return None

class GData(object):
    """Provides interface to Gkeyll output data.

    GData serves as a baseline interface to Gkeyll 1 and 2 output
    data. It allows for loading and saving and serves as an input for
    other Postgkyl methods. Represents a dataset in the Postgkyl
    command line mode.

    Init Args:
        fName (str): String with either full file name of or a
            fragment (for history loading).  Currently supports only
            'h5' and 'bp' files. Empty GData object is created when
            not specified
        stack (bool): Turn the internal stack on (True) and off
            (False) (default: True)
        comp (int or 'int:int'): Preselect a componend index (or a
            slice) for the partial load of data
        coord0 (int or 'int:int'): Preselect an index (or a slice) for
            the first coordinate for the partial load of data
        coord1 (int or 'int:int'): Preselect an index (or a slice) for
            the second coordinate for the partial load of data
        coord2 (int or 'int:int'): Preselect an index (or a slice) for
            the third coordinate for the partial load of data
        coord3 (int or 'int:int'): Preselect an index (or a slice) for
            the fourth coordinate for the partial load of data
        coord4 (int or 'int:int'): Preselect an index (or a slice) for
            the fifth coordinate for the partial load of data
        coord5 (int or 'int:int'): Preselect an index (or a slice) for
            the sixth coordinate for the partial load of data

    Raises:
        NameError: when file name does not exist or is empty/corrupted.
        TypeError: when partial load indices are no integers or 'int:int'

    Notes:
        - When fine name is incomplete or is not a Gkyl frame,
          Postgkyl tries opening it as a history data before throwing
          an exception
        - Preselection coordinate indices for higher dimension than
          included in the data are safelly disregarded (i.e., coord2,
          coord3, coord4, and coord5 for 2D data).
        - postgkyl.GData is a shortcut for postgkyl.data.GData

    Examples:
        import postgkyl
        data = postgkyl.GData('file.bp', comp=1)
    """

    def __init__(self, fName=None, stack=False, comp=None,
                 coord0=None, coord1=None, coord2=None,
                 coord3=None, coord4=None, coord5=None,
                 compgrid=False):
        self._stack = stack  # Turn OFF the stack
        self._compGrid = compgrid # disregard the mapped grid?
        self._grid = []  # cell centered grid
        self._gridType = "uniform" # type of grid
        self._gridFile = "grid" # name of grid file
        self._values = []  # (N+1)D narray of values 
        self.time = None
        self.frame = 0

        self.changeset = None
        self.builddate = None

        self.fName = fName
        if fName is not None:
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
    def _createOffsetCountBp(self, bpVar, coords, comp):
        numDims = len(bpVar.dims)
        count = np.array(bpVar.dims)
        offset = np.zeros(numDims, np.int)
        cnt = 0

        for d, coord in enumerate(coords):
            if d < numDims-1 and coord is not None:  # Last dim stores comp
                coord = idxParser(coord)
                if isinstance(coord, int):
                    offset[d] = coord
                    count[d] = 1
                elif isinstance(coord, slice):
                    offset[d] = coord.start
                    count[d] = coord.stop - coord.start
                else:
                    raise TypeError("'coord' is neither number or slice")
                cnt = cnt + 1
                
        if comp is not None:
            comp = idxParser(comp)
            if isinstance(comp, int):
                offset[-1] = comp
                count[-1] = 1
            elif isinstance(comp, slice):
                offset[d] = comp.start
                count[d] = comp.stop - comp.start
            else:
                raise TypeError("'comp' is neither number or slice")
            cnt = cnt + 1

        if cnt > 0:
            return tuple(offset), tuple(count)
        else:
            return (), ()

    def _loadFrame(self, axes=(None, None, None, None, None, None),
                   comp=None):
        extension = self.fName.split('.')[-1]
        # Gkeyll HDF5 frame load
        if extension == 'h5':
            fh = tables.open_file(self.fName, 'r')
            if not '/StructGridField' in fh:
                fh.close()
                self._loadSequence()
                return
            
            # Get the atributes
            lower = fh.root.StructGrid._v_attrs.vsLowerBounds
            upper = fh.root.StructGrid._v_attrs.vsUpperBounds
            cells = fh.root.StructGrid._v_attrs.vsNumCells
            # Load data ...
            self._values.append(fh.root.StructGridField.read())
            # ... and the time-stamp
            if '/timeData' in fh:
                self.time = fh.root.timeData._v_attrs.vsTime

            fh.close()

        # Gkyl ADIOS frame load
        elif extension == 'bp':
            fh = adios.file(self.fName)
            if not 'CartGridField' in fh.vars:
                # Not a Gkyl "frame" data; trying to load as a sequence
                fh.close()
                self._loadSequence()  
                return

            # Get the atributes
            # Postgkyl conventions require the atribuest to be
            # narrays even for 1D data
            lower = np.atleast_1d(adios.attr(fh, 'lowerBounds').value)
            upper = np.atleast_1d(adios.attr(fh, 'upperBounds').value)
            cells = np.atleast_1d(adios.attr(fh, 'numCells').value)
            if 'changeset' in fh.attrs.keys():
                self.changeset = adios.attr(fh, 'changeset').value.decode('UTF-8')
            if 'builddate' in fh.attrs.keys():
                self.builddate = adios.attr(fh, 'builddate').value.decode('UTF-8')

            # Load data ...
            var = adios.var(fh, 'CartGridField')
            offset, count = self._createOffsetCountBp(var, axes, comp)
            self._values.append(var.read(offset=offset, count=count))
            # ... and the time-stamp
            if 'time' in fh.vars:
                self.time = adios.var(fh, 'time').read()
            if 'frame' in fh.vars:
                self.frame = adios.var(fh, 'frame').read()

            # Check for mapped grid ...
            if "type" in fh.attrs.keys() and self._compGrid is False:
                self._gridType = adios.attr(fh, "type").value.decode('UTF-8')
            # .. load nodal grid if provided ...
            if self._gridType == "uniform":
                pass # nothing to do for uniform grids
            elif self._gridType == "mapped":
                if "grid" in fh.attrs.keys():
                    gridNm = adios.attr(fh, "grid").value.decode('UTF-8')
                else:
                    gridNm = "grid"  
                with adios.file(gridNm) as gridFh:
                    gridVar = adios.var(gridFh, 'CartGridField')
                    tmp = gridVar.read(offset=offset, count=count)
                    grid = [tmp[..., d].transpose() 
                            for d in range(tmp.shape[-1])]
                    self._grid.append(grid)
            elif self._gridType == "nonuniform":
                raise TypeError("'nonuniform' is not presently supported")
            else:
                raise TypeError("Unsupported grid type info in field!")

            # Adjust boundaries for 'offset' and 'count'
            numDims = len(cells)
            dz = (upper - lower) / cells
            if offset:
                if self._gridType == "uniform":
                    lower = lower + offset*dz
                    cells = cells - np.int(offset)
                elif self._gridType == "mapped":
                    for d in range(numDims):
                        idx = np.full(numDims, offset[d])
                        lower[d] = self._gridND[0][idx, d]
                        cells[d] = cells[d] - np.int(offset[d])
            if count:
                if self._gridType == "uniform":
                    upper = lower + count*dz
                    cells = np.int(count)
                elif self._gridType == "mapped":
                    for d in range(numDims):
                        idx = np.full(numDims, offset[d]+count[d])
                        upper[d] = self._gridND[0][idx ,d]
                        cells[d] = np.int(count[d])

            fh.close()
                            
        else:
            raise NameError(
                "File extension '{:s}' is not supported".
                format(extension))

        numDims = len(cells)
        dz = (upper - lower) / cells
        # Adjusts bounds in case ghost layer is included in data
        for d in range(numDims):
            if cells[d] != self._values[0].shape[d]:
                if self._gridType == "mapped":
                    raise ValueError("Data appears to include ghost cells which is not compatible with mapped grid. Use computational grid 'compgrid' instead.")
                ngl = int(np.floor((cells[d] - self._values[0].shape[d])*0.5))
                ngu = int(np.ceil((cells[d] - self._values[0].shape[d])*0.5))
                cells[d] = self._values[0].shape[d]
                lower[d] = lower[d] - ngl*dz[d]
                upper[d] = upper[d] + ngu*dz[d]

        # Construct grids if not loaded already
        if len(self._grid) == 0:
            grid = [np.linspace(lower[d],
                                upper[d],
                                cells[d]+1)
                    for d in range(numDims)]
            self._grid.append(grid)

    def _loadSequence(self):
        # Sequence load typically cancatenates multiple files
        files = glob('{:s}*'.format(self.fName))
        if not files:
            raise NameError(
                "File(s) '{:s}' not found or empty.".
                format(self.fName))

        cnt = 0  # Counter for the number of loaded files
        for fName in files:
            extension = fName.split('.')[-1]
            # Gkeyll HDF5 history load
            if extension == 'h5':
                fh = tables.open_file(fName, 'r')
                if '/DataStruct/data' in fh and \
                   '/DataStruct/timeMesh' in fh:
                    grid = fh.root.DataStruct.timeMesh.read()
                    values = fh.root.DataStruct.data.read()
                    fh.close()
                else:
                    fh.close()
                    continue

            # Gkyl ADIOS history load
            elif extension == 'bp':
                fh =  adios.file(fName)
                if 'Data' in fh.vars and 'TimeMesh' in fh.vars:
                    values = adios.var(fh, 'Data').read()
                    grid = adios.var(fh, 'TimeMesh').read()
                    fh.close()
                else:
                    fh.close()
                    continue
            else:
                continue
                        
            if cnt > 0:
                self._grid[0] = np.append(self._gridCC[0], grid, axis=0)
                self._values[0] = np.append(self._values[0], values, axis=0)
            else:
                self._grid.append(grid)
                self._values.append(values)
            cnt += 1

        if cnt == 0:  # No files loaded
            raise NameError(
                "File(s) '{:s}' not found or empty.".
                format(self.fName))

        # Squeeze the time coordinate ...
        if len(self._grid[0].shape) > 1:
            self._grid[0] = np.squeeze(self._grid[0])
        # ... and make it a list following the Postgkyl grid conventions
        self._grid[0] = [self._grid[0]]

        # glob() doesn't guarantee the right order
        sortIdx = np.argsort(self._grid[0][0])
        self._grid[0][0] = self._grid[0][0][sortIdx]
        self._values[0] = self._values[0][sortIdx, ...]


    #-----------------------------------------------------------------
    #-- Stack Control ------------------------------------------------
    def getNumCells(self):
        if len(self._values) > 0:
            numDims = len(self._values[-1].shape)-1
            cells = np.zeros(numDims)
            for d in range(numDims):
                cells[d] = self._values[-1].shape[d]
            return cells
        else:
            return 0

    def getNumComps(self):
        if len(self._values) > 0:
            return self._values[-1].shape[-1]
        else:
            return 0

    def getNumDims(self, squeeze=False):
        if len(self._values) > 0:
            numDims = len(self._values[-1].shape)-1
            if squeeze:
                cells = self.getNumCells()
                for d in range(numDims):
                    if cells[d] == 1:
                        numDims = numDims - 1
            return numDims
        else:
            return 0

    def getBounds(self):
        if len(self._grid) > 0:
            numDims = len(self._values[-1].shape)-1
            lo, up = np.zeros(numDims), np.zeros(numDims)
            for d in range(numDims):
                lo[d] = self.grid[-1][d].min()
                up[d] = self.grid[-1][d].max()
            return lo, up
        else:
            return np.array([]), np.array([])

    def getGrid(self, nodal=False, compSpace=False):
        if len(self._values) > 0:
            if not nodal:
                grid = self._gridCC[-1]
            else:
                grid = self._gridND[-1]
                return grid
        else:
            return []

    def getValues(self):
        if len(self._values) > 0:
            return self._values[-1]
        else:
            return np.array([])


    def popGrid(self, nodal=False):
        if len(self._grid) > 0:
            self._lower.pop()
            self._upper.pop()
            self._cells.pop
            if not nodal:
                self._gridND.pop()
                return self._gridCC.pop()
            else:
                self_.gridCC.pop()
                return self._gridND.pop()
        else:
            return []
    def popValues(self):
        if len(self._values) > 0:
            return self._values.pop()
        else:
            return np.array([])

    # def pushBoundsAndCells(self, lower, upper, cells):
    #     if not self._stack:
    #         self._lower.append(lower)
    #         self._upper.append(upper)
    #         self._cells.append(cells)
    #     else:
    #         self._lower[0] = lower
    #         self._upper[0] = upper
    #         self._cells[0] = cells

    def pushGrid(self, grid, lower=None, upper=None, nodal=False):
        lo, up = self.getBounds()
        cells = np.zeros(self.getNumDims())
        for d in range(self.getNumDims()):
            cells[d] = len(grid[d])
        if not self._stack:
            if not nodal:
                self._gridCC.append(grid)
            else:
                self._gridND.append(grid)
            self._cells.append(cells)

            if lower is not None:
                self._lower.append(lower)
            else:
                self._lower.append(lo)
            if upper is not None:
                self._upper.append(upper)
            else:
                self._upper.append(up)
        else:
            if not nodal:
                self._gridCC[0] = grid
            else:
                self._gridND[0] = grid
            self._cells[0] = cells

            if lower is not None:
                self._lower[0] = lower
            if upper is not None:
                self._upper[0] = upper
    def pushValues(self, values):
        if not self._stack:
            self._values.append(values)
        else:
            self._values[0] = values


    # legacy function
    def peakGrid(self):
        return self.getGrid()
    def peakValues(self):
        return self.getValues()

    #-----------------------------------------------------------------
    #-- Info ---------------------------------------------------------
    def info(self):
        """Prints GData object information.

        Prints time (only when available), number of components, dimension
        spans, extremes for a GData object. Only the top of the stack is
        printed.
        
        Args:
            none
        
        Returns:
            output (str): A list of strings with the informations
        """
        values = self.peakValues()
        numComps = self.getNumComps()
        numDims = self.getNumDims()
        numCells = self.getNumCells()
        lower, upper = self.getBounds()

        if len(values) > 0:
            maximum = values.max()
            maxIdx = np.unravel_index(np.argmax(values), values.shape)
            minimum = values.min()
            minIdx = np.unravel_index(np.argmin(values), values.shape)
        
            output = ""

            if self.time is not None:
                output += "- Time: {:e}\n".format(self.time)
            if self.frame is not None:
                output += "- Frame: {:d}\n".format(self.frame)
            output += "- Number of components: {:d}\n".format(numComps)
            output += "- Number of dimensions: {:d}\n".format(numDims)
            output += "- Grid: ({:s})\n".format(self._gridType)
            for d in range(numDims):
                output += "  - Dim {:d}: Num. cells: {:d}; ".format(d, int(numCells[d]))
                output += "Lower: {:e}; Upper: {:e}\n".format(lower[d],
                                                              upper[d])
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

            if self.changeset is not None and self.builddate is not None:

                output += "\n- Created with Gkeyll:\n"
                output += "  - Changeset: {:s}\n".format(self.changeset)
                output += "  - Build Date: {:s}".format(self.builddate)
            return output
        else:
            return "No data"

    #-----------------------------------------------------------------
    #-- Write --------------------------------------------------------
    def write(self, bufferSize=1000, fName=None, txt=False):
        """Writes data in ADIOS .bp file or ASCII .txt file
        """
        if txt:
            mode = 'txt'
        else:
            mode = 'bp'
        # Create output file name
        if fName is None:
            if self.fName is not None:
                fn = self.fName
                fName = fn.split('.')[0].strip('_') + '_mod.' + mode
            else:
                fName = "gdata." + mode
        else:
            if isinstance(fName, str):
                raise TypeError("'fName' must be a string")
            if fName.split('.')[-1] != mode:
                fileName += '.' + mode

        numDims = self.getNumDims()
        numComps = self.getNumComps()
        numCells = self.getNumCells()

        if mode == 'bp':
            # Create string number of cells and offsets
            sNumCells = ""
            sOffsets = ""
            for i in range(numDims):
                sNumCells += "{:d},".format(numCells[i])
                sOffsets += "0,"
            sNumCells += "{:d}".format(numComps)
            sOffsets += "0"

            # ADIOS init
            adios.init_noxml()
            adios.set_max_buffer_size(bufferSize)
            groupId = adios.declare_group("CartField", "")
            adios.select_method(groupId, "POSIX1", "", "")

            # Define variables and attributes
            adios.define_attribute_byvalue(groupId, "numCells", "", numCells)
            lo, up = self.getBounds()
            adios.define_attribute_byvalue(groupId, "lowerBounds", "", lo)
            adios.define_attribute_byvalue(groupId, "upperBounds", "", up)
            if self.time is not None:
                adios.define_var(groupId, "time", "",
                                 adios.DATATYPE.double, "", "", "")
            adios.define_var(groupId, "CartGridField", "",
                             adios.DATATYPE.double,
                             sNumCells, sNumCells, sOffsets)

            # Write the data and finalize
            fh = adios.open("CartField", fName, 'w')
            if self.time is not None:
                adios.write(fh, "time", self.time)
            adios.write(fh, "CartGridField", self.peakValues())
            adios.close(fh)
            adios.finalize()

            # Cheating
            if len(fName.split('/')) > 1:
                nm = fName.split('/')[-1]
            else:
                nm = fName
            shutil.move(fName + '.dir/' + nm + '.0', fName)
            shutil.rmtree(fName + '.dir')

        elif mode == 'txt':
            numRows = int(numCells.prod())
            grid = self.peakGrid()
            values = self.peakValues()

            basis = np.full(numDims, 1.0)
            for d in range(numDims-1):
                basis[d] = numCells[(d+1):].prod()

            fh = open(fName, 'w')
            for i in range(numRows):
                idx = i
                idxs = np.zeros(numDims, np.int)
                for d in range(numDims):
                    idxs[d] = int(idx // basis[d])
                    idx = idx % basis[d]

                line = ""
                for d in range(numDims-1):
                    line += "{:e}, ".format(grid[d][idxs[d]])
                line += "{:e}; ".format(grid[numDims-1][idxs[numDims-1]])
                for c in range(numComps-1):
                    line += "{:e}, ".format(values[tuple(idxs)][c])
                line += "{:e}\n".format(values[tuple(idxs)][numComps-1])
                fh.write(line)
            fh.close()

