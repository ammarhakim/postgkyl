#!/usr/bin/env python
"""
Postgkyl script to plot data directly from the terminal
"""
# standart imports
import numpy
import matplotlib.pyplot as plt
import exceptions
import sys
import os
from optparse import OptionParser
# custom imports
import postgkyl as pg

#---------------------------------------------------------------------
# Parser -------------------------------------------------------------
parser = OptionParser()
# what to plot
parser.add_option('-p', '--plot', action = 'store',
                  dest = 'fName',
                  help = 'G file to plot')
parser.add_option('-y', '--history', action = 'store',
                  dest = 'fNameRoot',
                  help = 'G history file root to plot')
parser.add_option('-c', '--component', action = 'store',
                  dest = 'component', default = 0,
                  help = 'Component to plot (default 0)')
parser.add_option('-m', '--mask', action = 'store',
                  dest = 'maskName',
                  help = 'G file that serves as a mask')
# projecting
parser.add_option('--ns', action = 'store',
                  dest = 'nodalSerendipity',
                  help = 'Polynomial order of the nodal Serendipity basis')
parser.add_option('--ms', action = 'store',
                  dest = 'modalSerendipity',
                  help = 'Polynomial order of the modal Serendipity basis')
parser.add_option('--mo', action = 'store',
                  dest = 'maxOrder',
                  help = 'Polynomial order of the max order basis')
# saving plot
parser.add_option('-s', '--save', action = 'store_true',
                  dest = 'save',
                  help = 'Save the displayed plot (png by default)')
parser.add_option('-o', '--output', action = 'store',
                  dest = 'outName',
                  help = 'When saving figures, use this file name')
# how to plot
parser.add_option('--xlabel', action = 'store',
                  dest = 'xlabel', default = '',
                  help = 'x-label to put on plots')
parser.add_option('--ylabel', action = 'store',
                  dest = 'ylabel', default = '',
                  help = 'y-label to put on plots')
parser.add_option('-t', '--title', action = 'store',
                  dest = 'title',
                  help = 'Set title to put on plots')
parser.add_option('--no-title', action = 'store_false',
                  dest = 'titleOn', default=True,
                  help = 'Turn OFF title to put on plots')
parser.add_option('-g', '--no-grid', action = 'store_false',
                  dest = 'gridOn', default = True,
                  help = 'Turn OFF the grid')
parser.add_option('--cmap', action = 'store',
                  dest = 'cmap', default = 'jet',
                  help = 'Color map to use for 2D plots (default \'jet\')')
parser.add_option('--axis-free', action = 'store_true',
                  dest = 'freeAxis',
                  help = "If set, 2D plots will no longer have equal axis",
                  default = False)
parser.add_option('--color', action = 'store',
                  dest = 'color', default = 'RoyalBlue',
                  help = "Color of 1D plots")
parser.add_option('--contour', action = 'store_true',
                  dest = 'contour', default = False,
                  help = "Plot contour instead of a bitmat for 2D plots")
# misellaneous
parser.add_option('--dont-show', action = 'store_true',
                  dest = 'dontShow', default = False,
                  help = 'Do not show plot')
parser.add_option('-x', '--xkcd', action = 'store_true',
                  dest = 'xkcd', default = False,
                  help = 'Plot xkcd.com style plots!')
parser.add_option('-w', '--write-history', action = 'store_true',
                  dest = 'writeHistory', default = False,
                  help = 'Write the loaded history to text a file')
(options, args) = parser.parse_args()

#---------------------------------------------------------------------
# Data Loading -------------------------------------------------------
def _centeredLinspace(lower, upper, numElem):
    dx = (upper-lower)/numElem
    return numpy.linspace(lower+0.5*dx, upper-0.5*dx, numElem)

# loading data to plot
if options.fName:
    data = pg.GData(options.fName)
    if options.nodalSerendipity:
        dg = pg.GInterpNodalSerendipity(data, int(options.nodalSerendipity))
        coords, values = dg.project(int(options.component))
        numDims = data.numDims
    elif options.modalSerendipity:
        dg = pg.GInterpModalSerendipity(data, int(options.modalSerendipity))
        coords, values = dg.project(int(options.component))
        numDims = data.numDims
    elif options.maxOrder:
        dg = pg.GInterpModalMaxOrder(data, int(options.maxOrder))
        coords, values = dg.project(int(options.component))
        numDims = data.numDims
    else:
        c = [_centeredLinspace(data.lowerBounds[d],
                               data.upperBounds[d],
                               data.numCells[d])
             for d in range(data.numDims)]
        coords = numpy.meshgrid(*c, indexing='ij')
        values = data.q[..., int(options.component)]
        numDims = data.numDims
elif options.fNameRoot:
    hist = pg.GHistoryData(options.fNameRoot)
    coords = numpy.expand_dims(hist.time, axis=0)
    values = hist.values
    numDims = 1
else:
    print(' *** No data specified for plotting')
    sys.exit()

# masking
if options.maskName:
    maskField = pg.GData(options.maskName).q[...,0]
    values = numpy.ma.masked_where(maskField < 0.0, values)

#---------------------------------------------------------------------
# Creating Titles and Names ------------------------------------------
if options.fName:
    name = options.fName
elif options.fNameRoot:
    name = options.fNameRoot

if options.fName:
    name = name.split('/')[-1] # get rid of the full path
    name = ''.join(name.split('.')[: -1]) # get rid of the extension
    # This weird Python construct is here in case someone would like
    # to use '.' in name... I really dislike it but I don't know about
    # any better -pc
    
    # add component number
    name = '{}_c{:d}'.format(name, int(options.component))
else:
   pass

if options.outName is None:
    outName = '{}/{}.png'.format(os.getcwd(), name)
else:
    outName = str(options.outName)

if options.title is None:
    if options.fName:
        title = '{}\nt={:1.2e}'.format(name, data.time)
    else:
        title = '{}\nhistory'.format(name)
else:
    title = str(options.title)
#---------------------------------------------------------------------
# Plotting -----------------------------------------------------------

# plotting parameters are based solely on the personal taste of Ammar :)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 16
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'large'
#plt.rcParams['xtick.major.size'] = 8 # default is 4
#plt.rcParams['xtick.major.width'] = 3 # default is 0.5
#plt.rcParams['ytick.major.size'] = 8 # default is 4
#plt.rcParams['ytick.major.width'] = 3 # default is 0.5
plt.rcParams['figure.facecolor'] = 'white'
#plt.rcParams['figure.subplot.bottom'] = 0.125
#plt.rcParams['figure.subplot.right'] = 0.85 # keep labels/ticks of
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['contour.negative_linestyle'] = 'solid'
#plt.rcParams['savefig.bbox'] = 'tight'
#plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['axes.titlesize'] = 10

# this needs to be set after the rest of rcParams
if options.xkcd:
    plt.rcParams['mathtext.default'] = 'regular'
    plt.xkcd()

# plot 
fig, ax = plt.subplots()
if numDims == 1:
    if not options.xkcd:
        im = ax.plot(coords[0], values, color=options.color)
    else:
        im = ax.plot(coords[0], values, color=options.color, 
                     clip_on=False, zorder=100)
elif numDims == 2:
    plt.set_cmap(options.cmap)
    if not options.contour:
        im = ax.pcolormesh(coords[0], coords[1], values)
    else:
        im = ax.contour(coords[0], coords[1], values)
else:
    raise exceptions.RuntimeError(
        "Plotting 3D data is not currently supported")

# format
def _colorbar(obj, _ax, _fig, redraw=False, aspect=None, label=''):
    """Add a colorbar adjacent to obj, with a matching height

    For use of aspect, see:
    http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_aspect
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #_fig_ = obj.figure
    #_ax_ = obj.axes
    _divider_ = make_axes_locatable(_ax)
    _cax_ = _divider_.append_axes("right", size="5%", pad=0.05)
    _cbar_ =  fig.colorbar(obj, cax=_cax_, label=label)
    if aspect != None:
        _ax.set_aspect(aspect)
    if redraw:
        _fig.canvas.draw()
    return _cbar_

if options.titleOn:
    ax.set_title(title)
ax.set_xlabel(str(options.xlabel))
ax.set_ylabel(str(options.ylabel))
ax.grid(options.gridOn)
if numDims == 1:
    plt.autoscale(enable=True, axis='x', tight=True)
    ax.axis('tight')
elif numDims == 2:
    _colorbar(im, ax, fig)
    if options.freeAxis:
        ax.axis('tight')
    else:
        ax.axis('image')

plt.tight_layout()

# this should be last formatting option
if options.xkcd:
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Turn OFF grid
    ax.grid(False)

if options.save:
    fig.savefig(outName, bbox_inches='tight')

if options.writeHistory:
    hist.save()

if not options.dontShow:
    plt.show()
else:
    plt.close(fig)
