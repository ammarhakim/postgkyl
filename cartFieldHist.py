#!/usr/bin/env python
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
    r"""
    numpy array that encapsules time history of multiple CartFields.
    Allows for batch manipulation and plotting.
    """

    def __init__(self, fileNameBase=None):
        self.isLoaded = 0
        if fileNameBase is not None:
            self.load(fileNameBase)

    def load(self, fileNameBase):
        r"""
        Load a batch of Gkeyll output files specified by 'fileNameBase'.
        """
        self.fileNameBase = fileNameBase
        files = glob.glob('{}_*.h5'.format(self.fileNameBase))
        if files == '':
            raise exceptions.RuntimeError(
                'CartFieldHist:load Files with base \'{}\' does not exist!'.format(self.fileNameBase))
        self.isLoaded = 1
        #self.history = numpy.zeros(len(files))
        time = numpy.zeros(len(files))
        self.history = [cartField.CartField(name) for name in files]
        self.history = numpy.array(self.history)

       #for i in range(len(files)):
       #     name = '{}_{}.h5'.format(self.fileNameBase, i)
       #     print(name)
       #     self.history[i] = cartField.CartField(name)
       #     time[i] = self.history[i].time
       # idxSort = numpy.argsort(time)
       # self.history = self.history(idxSort)
