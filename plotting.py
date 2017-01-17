#!/usr/bin/env python
"""
postgkyl module containing plotting methods
"""

# standart imports
import numpy
import matplotlib.pyplot as plt                      
# custom imports
import cartField

# Helper functions
def figureSetup():
    """Set basic figure parameters.

    Inputs:
    None

    Keyword arguments:
    None

    Returns:

    Note:
    Plotting parameters are based solely on the personal taste of Ammar :)
    """
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
    r"""
    Add a colorbar adjacent to obj, with a matching height
    For use of aspect, see http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_aspect ; E.g., to fill the rectangle, try "auto"
    """
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

def plot1D(dataX, dataY, ax=None,
           color='RoyalBlue'):
    """Plot 1D field  data

    Inputs:
    dataX -- self-explanatory (I hope...)
    dataY -- self-explanatory

    Keyword arguments:
    ax    -- specify an axes to plot in (default None - new figure is created)
    color -- line color (default 'RoyalBlue')
    
    Returns:
    None
    """
    # set the basic plotting parameters 
    figureSetup()

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    out = ax.plot(dataX, dataY, color=color)
    print(out)
    ax.set_xlim([numpy.min(dataX), numpy.max(dataX)])
    ax.grid(True)
    plt.tight_layout(True)

def plot2D(dataX, dataY, dataZ, ax=None):
    """Plot 2D field  data

    Inputs:
    dataX -- self-explanatory (I hope...)
    dataY -- self-explanatory
    dataZ -- self-explanatory

    Keyword arguments:
    ax    -- specify an axes to plot in (default None - new figure is created)
    
    Returns:
    None
    """
    # set the basic plotting parameters
    figureSetup()

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    out = ax.pcolormesh(dataX, dataY, dataZ)
    print(out)
    ax.axis('tight')
    #colorbar(ax)

def plotField(field, comp=0, isDG=False,
              fix1=None, fix2=None, fix3=None,
              fix4=None, fix5=None, fix6=None,
              ax=None,
              color='RoyalBlue'):
    """Plot the data of specified field.

    Inputs:
    field -- CartField or CartFieldDG object

    Keyword arguments:
    comp -- list or tuple of components to be plotted (default 0)
    isDG -- specifies if to look for projected data
    fix1 -- fixes the first coordinate to provided index (default None)
    fix2 -- fixes the second coordinate to provided index (default None)
    fix3 -- fixes the third coordinate to provided index (default None)
    fix4 -- fixes the fourth coordinate to provided index (default None)
    fix5 -- fixes the fifth coordinate to provided index (default None)
    fix6 -- fixes the sixth coordinate to provided index (default None)

    Returns:
    None

    Note: 
    If the projected data re not available and 'isDG' flag is on,
    the project method of the field will be called before plotting.

    Exceptions:
    RuntimeError -- raised when no data have been loaded yet
    """
    if not field.isLoaded:
        raise exceptions.RuntimeError(
            "plotting.plotField: Data needs to be loaded first. Use CartField.load(fileName).")

    if isDG and not field.isProj:
        print("plotting.plotField: Data not projected, projecting.")
        field.project()

    if not isDG:
        coords = field.coords
    else:
        coords = field.coordsProj
        
    comp = numpy.array(comp)
    if comp.ndim == 0:
        comp = numpy.expand_dims(comp, 0)
    for i in range(comp.size):
        # select propper data
        if not isDG:
            values = field.q
        else:
            values = field.qProj
        # select component
        if len(values.shape) != field.numDims:
            mask    = numpy.zeros(values.shape[field.numDims])
            mask[i] = 1
            values  = numpy.compress(mask, values, field.numDims)
            values  = numpy.squeeze(values)
        coordsPlot, valuesPlot = cartField.fixCoordinates(coords, values,
                                                          fix1, fix2, fix3,
                                                          fix4, fix5, fix6)
        # squeeze degenerated dimensions
        coordsPlot = numpy.squeeze(coordsPlot)
        valuesPlot = numpy.squeeze(valuesPlot)
        
        if len(valuesPlot.shape) == 1:
            plot1D(coordsPlot, valuesPlot,
                   ax=ax,
                   color=color)
        elif len(valuesPlot.shape) == 2:
            plot2D(numpy.transpose(coordsPlot[0]),
                   numpy.transpose(coordsPlot[1]),
                   numpy.transpose(valuesPlot),
                   ax=ax)
        else:
            raise exeptions.RuntimeError(
                "plotting.plotField: Dimension of the field is bigger than two. Some dimensions need to be fixed.") 


def plotFieldHist(fieldHist, numSnapshots,
                  comp=0, isDG=False,
                  fix1=None, fix2=None, fix3=None,
                  fix4=None, fix5=None, fix6=None,
                  ax=None):
    """Plot snapshots of the data in specified field history object.

    Inputs:
    fieldHist    -- CartFieldHist or CartFieldDGHist object
    numSnapshots -- number of time snapshots to create

    Keyword arguments:
    comp -- list or tuple of components to be plotted (default 0)
    isDG -- specifies if to look for projected data
    fix1 -- fixes the first coordinate to provided index (default None)
    fix2 -- fixes the second coordinate to provided index (default None)
    fix3 -- fixes the third coordinate to provided index (default None)
    fix4 -- fixes the fourth coordinate to provided index (default None)
    fix5 -- fixes the fifth coordinate to provided index (default None)
    fix6 -- fixes the sixth coordinate to provided index (default None)

    Returns:
    None

    Note: 
    If the projected data re not available and 'isDG' flag is on,
    the project method of the field will be called before plotting.

    Exceptions:
    RuntimeError -- raised when no data have been loaded yet
    """
    # get colormap
    mask = plt.get_cmap('coolwarm')
    
    ## create shanpshot indices
    snapshots = numpy.linspace(0, len(fieldHist.history)-1,
                               numSnapshots)
    # convert array to nearest integers
    snapshots = numpy.rint(snapshots)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    for i, idx in enumerate(snapshots):
        color = mask(float(i)/(numSnapshots-1))
        plotField(fieldHist.history[int(idx)], comp=comp, isDG=isDG,
                  fix1=fix1, fix2=fix2, fix3=fix3,
                  fix4=fix4, fix5=fix5, fix6=fix6,
                  ax=ax,
                  color=color)
