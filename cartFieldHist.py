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
        r"""
        Plots the specified number of snapshots.
        """
        plotting.plotFieldHist(self, numSnapshots, comp=comp,
                               fix1=fix1, fix2=fix2, fix3=fix3,
                               fix4=fix4, fix5=fix5, fix6=fix6)
