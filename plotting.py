#!/usr/bin/env python
r"""
Basic Gkeyll plotting methods
"""

# standart imports
import numpy
import matplotlib.pyplot as plt

# Helper plotting methods
def figureSetup():
    plt.rcParams['lines.linewidth']            = 4
    plt.rcParams['font.size']                  = 18
    #plt.rcParams['font.weight']                = 'bold'
    plt.rcParams['axes.labelsize']             = 'large'
    #plt.rcParams['xtick.major.size']           = 8 # default is 4
    #plt.rcParams['xtick.major.width']          = 3 # default is 0.5
    #plt.rcParams['ytick.major.size']           = 8 # default is 4
    #plt.rcParams['ytick.major.width']          = 3 # default is 0.5
    plt.rcParams['figure.facecolor']           = 'white'
    #plt.rcParams['figure.subplot.bottom']      = 0.125
    #plt.rcParams['figure.subplot.right']       = 0.85 # keep labels/ticks of
    #colobar in figure
    plt.rcParams['image.interpolation']        = 'none'
    plt.rcParams['image.origin']               = 'lower'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    #plt.rcParams['savefig.bbox']               = 'tight'
    #plt.rcParams['mathtext.default'] = 'regular'

def colorbar(obj, mode=1, redraw=False, _fig_=None, _ax_=None, aspect=None):
    '''
    Add a colorbar adjacent to obj, with a matching height
    For use of aspect, see http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_aspect ; E.g., to fill the rectangle, try "auto"
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if mode == 1:
        _fig_ = obj.figure; _ax_ = obj.axes
    elif mode == 2: # assume obj is in the current figure/axis instance
        _fig_ = plt.gcf(); _ax_ = plt.gca()
    _divider_ = make_axes_locatable(_ax_)
    _cax_ = _divider_.append_axes("right", size="5%", pad=0.05)
    _cbar_ =  _fig_.colorbar(obj, cax=_cax_)
    if aspect != None:
        _ax_.set_aspect(aspect)
    if redraw:
        _fig_.canvas.draw()
    return _cbar_

def plot1D(dataX, dataY):
    r"""
    Plots 1D Gkeyll data
    """
    figureSetup()
    fig, ax = plt.subplots(1, 1)
    ax.plot(dataX, dataY)
    ax.grid()
    plt.tight_layout()

def plot2D(dataX, dataY, dataZ):
    r"""
    Plots 2D Gkeyll data
    """
    figureSetup()
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(dataX, dataY, dataZ)
    ax.axis('tight')
    #colorbar(ax)
     
 
