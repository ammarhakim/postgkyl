#!/usr/bin/env python
r"""
Basic Gkeyll plotting methods
"""

# standart imports
import numpy
import matplotlib.pyplot as plt

# Helper plotting methods
def fixCoordinates(coords, values,
                   fix1=None, fix2=None, fix3=None,
                   fix4=None, fix5=None, fix6=None):
    r"""
    Fixes specified coordinates and decreases the dimension of data.
    """
    fix = (fix1, fix2, fix3, fix4, fix5, fix6)
    coordsFix = coords
    valuesFix = values
    for i, value in reversed(list(enumerate(fix))):
        if value is not None and len(values.shape) > i:
            mask = numpy.zeros(values.shape[i])
            mask[fix[i]] = 1

            coordsFix = numpy.delete(coordsFix, i, 0)
            coordsFix = numpy.compress(mask, coordsFix, axis=i+1)  
            coordsFix = numpy.squeeze(coordsFix)

            valuesFix = numpy.compress(mask, valuesFix, axis=i) 
            valuesFix = numpy.squeeze(valuesFix)
    return coordsFix, valuesFix
                      

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

def plotField(field, comp=0, isDG=False,
              fix1=None, fix2=None, fix3=None,
              fix4=None, fix5=None, fix6=None):

    if not field.isLoaded:
        raise exceptions.RuntimeError(
            "CartField.plot: Data needs to be loaded first. Use CartField.load(fileName).")

    if isDG and not field.isProj:
        print("Data not projected, projecting.")
        field.project()

    if not isDG:
        coords = []
        for i in range(field.data.ndim):
            temp = numpy.linspace(field.data.lowerBounds[i], field.data.upperBounds[i], field.data.cells[i])
            coords.append(temp)
        coords = numpy.array(numpy.meshgrid(*coords, indexing='ij'))
    else:
        coords = field.coords
        
    comp = numpy.array(comp)
    if comp.ndim == 0:
        comp = numpy.expand_dims(comp, 0)
    for i in range(comp.size):
        # extractiong the coordinate
        if not isDG:
            values  = field.data.q
        else:
            values = field.dataProj
        if len(values.shape) != field.data.ndim:
            mask    = numpy.zeros(values.shape[field.data.ndim])
            mask[i] = 1
            values = numpy.compress(mask, values, field.data.ndim)
            values = numpy.squeeze(values)
        coordsPlot, valuesPlot = fixCoordinates(coords, values,
                                                fix1, fix2, fix3, fix4, fix5, fix6)
        if len(valuesPlot.shape) == 1:
            plot1D(coordsPlot, valuesPlot)
        elif len(valuesPlot.shape) == 2:
            plot2D(numpy.transpose(coordsPlot[0]),
                   numpy.transpose(coordsPlot[1]),
                   numpy.transpose(valuesPlot))
        else:
            raise exeptions.RuntimeError(
                "CartField.plot: Dimension of the field is bigger than two. Some dimensions need to be fixed.") 
