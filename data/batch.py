#!/usr/bin/env python
"""
Postgkyl sub-module for convinience batch manipulation
"""
import numpy
import glob
import sys
import exceptions

import data
import interp

class GBatchData:
    """Convenience batch load of GData

    __init__(fNameRoot : string)
    Search for files and load them to 'batch' array
    """

    def __init__(self, fNameRoot):
        """Search for files and load them to 'batch' array

        Parameters:
        fNameRoot -- file name roots

        Notes:
        Load function is determined based on the extension
        """
        
        self.fNameRoot = fNameRoot
        files = glob.glob('*{}*.??'.format(self.fNameRoot))
        if files == []:
            raise exceptions.RuntimeError(
                'GBatchData: Files with root \'{}\' do not exist!'.
                format(self.fNameRoot))

        # load batch
        self.array = [gData.GData(name) for name in files]
        self.array = numpy.array(self.array)
        # sort history
        time = [temp.time for temp in self.array]
        sortIdx = numpy.argsort(time)
        self.array = self.array[sortIdx]

class GBatchInterpNodalSerendipity:
    def __init__(self, data, polyOrder):
        self.data = data
        self.polyOrder = polyOrder

    def project(self, comp=0):
        projection = []
        for i, dat in enumerate(self.data.array):
            projObj = gInterp.GInterpNodalSerendipity(dat, self.polyOrder)
            coords, temp = projObj.project(comp)
            projection.append(temp)
            percent = float(i)/len(self.data.array)*100
            sys.stdout.write(
                '\rGBatchInterpNodalSerendipity projecting: {:6.2f}% done'.
                format(percent))
            sys.stdout.flush()
        print('\rGBatchInterpNodalSerendipity projecting: 100.00% done')
        return numpy.array(coords), numpy.array(projection)

class GBatchInterpModalSerendipity:
    def __init__(self, data, polyOrder):
        self.data = data
        self.polyOrder = polyOrder

    def project(self, comp=0):
        projection = []
        for i, dat in enumerate(self.data.array):
            projObj = gInterp.GInterpModalSerendipity(dat, self.polyOrder)
            coords, temp = projObj.project(comp)
            projection.append(temp)
            percent = float(i)/len(self.data.array)*100
            sys.stdout.write(
                '\rGBatchInterpModalSerendipity projecting: {:6.2f}% done'.
                format(percent))
            sys.stdout.flush()
        return coords, projection

class GBatchInterpModalMaxOrder:
    def __init__(self, data, polyOrder):
        self.data = data
        self.polyOrder = polyOrder

    def project(self, comp=0):
        projection = []
        for i, dat in enumerate(self.data.array):
            projObj = gInterp.GInterpModalMaxOrder(dat, self.polyOrder)
            coords, temp = projObj.project(comp)
            projection.append(temp)
            percent = float(i)/len(self.data.array)*100
            sys.stdout.write(
                '\rGBatchInterpModalMaxOrder projecting: {:6.2f}% done'.
                format(percent))
            sys.stdout.flush()
        return coords, projection
