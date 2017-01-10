r"""Read HDF files produced by Gkeyll
"""

import numpy
import tables
import os
import exceptions

class GkeData:
    r"""GkeData(fName: string) -> GkeData

    Provides an interface to read data stored in a Gkeyll produced
    HDF5 file names 'fName'.
    """

    def __init__(self, fName):
        self.fName = fName
        if not os.path.exists(self.fName):
            raise exceptions.RuntimeError(
                "GkeData: File %s does not exist!" % fName)
        self.fh = tables.openFile(fName)

        grid = self.fh.root.StructGrid
        # read in information about grid
        self.lowerBounds = grid._v_attrs.vsLowerBounds
        self.upperBounds = grid._v_attrs.vsUpperBounds
        self.cells = grid._v_attrs.vsNumCells
        self.ndim = len(self.cells)

        # read in time data if it exists
        try:
            self.time = self.fh.root.timeData._v_attrs.vsTime
        except:
            self.time = 0.0
        
        # read in data
        self.q = self.fh.root.StructGridField

    def close(self):
        r"""close() -> None

        Closes the file
        """
        self.fh.close()
        
class GkeHistoryData:
    r"""GkeHistoryData(base: string, [component : int, start : int]) -> GkeHistoryData

    Given a base name to a history, reads in all existing frames and
    concatenates them into a single array. Optionally, a ``start``
    frame can be specified and the history will be loaded starting
    from that frame.

    Once the class is constructed the time series can be accessed
    using the ``history`` field and the times when these were taken
    using the ``time`` field.
    """

    def __init__(self, base, start=0):
        self.base = base

        # read in first history file
        fn = base + ("_%d.h5" % start)
        fh = tables.openFile(fn)
        self.history = fh.root.DataStruct.data.read()
        self.time = fh.root.DataStruct.timeMesh.read()
        fh.close()

        # now keep loading data till we run of frames
        currFrame = start+1
        while True:
            fn = base + ("_%d.h5" % currFrame)
            if not os.path.exists(fn):
                break
            fh = tables.openFile(fn)
            self.history = numpy.append(self.history, fh.root.DataStruct.data.read(), axis=0)
            self.time = numpy.append(self.time, fh.root.DataStruct.timeMesh.read())
            currFrame = currFrame + 1
            fh.close()

