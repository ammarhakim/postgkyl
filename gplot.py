#!/usr/bin/env python
"""
Postgkyl script to plot data directly from the terminal
"""
# standart imports
import numpy
import matplotlib.pyplot as plt
import exceptions
import sys
from optparse import OptionParser
# custom imports
import postgkyl as pg

# set terminal options
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
                  dest = 'savePng',
                  help = 'Save png of plot displayed')
parser.add_option('-o', '--output', action = 'store',
                  dest = 'outNm',
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
                  help = 'Title to put on plots')
parser.add_option('-g', '--grid', action = 'store_false',
                  dest = 'grid', default = True,
                  help = 'Do not show grid')
parser.add_option('--cmap', action = 'store',
                  dest = 'cmap',
                  help = 'Color map to use for 2D plots')
parser.add_option('--axis-eq', action = 'store_true',
                  dest = 'eqvAxis',
                  help = "If set, 2D plots will have equal axis",
                  default = False)
parser.add_option('--color', action = 'store',
                  dest = 'color', default = 'RoyalBlue',
                  help = "Color of 1D plots")
# misellaneous
parser.add_option('--dont-show', action = 'store_true',
                  dest = 'dontShow', default = False,
                  help = 'Do not show plot')
parser.add_option('-x', '--xkcd', action = 'store_true',
                  dest = 'xkcd', default = False,
                  help = 'Plot xkcd.com style plots!')

(options, args) = parser.parse_args()

def centeredLinspace(lower, upper, numElem):
    dx = (upper-lower)/numElem
    return numpy.linspace(lower+0.5*dx, upper-0.5*dx, numElem)

# loading data to plot
if options.fName:
    data = pg.GData(options.fName)
    if options.nodalSerendipity:
        dg = pg.GInterpNodalSerendipity(data, int(options.nodalSerendipity))
        coords, values = dg.project(options.component)
        numDims = data.numDims
    elif options.modalSerendipity:
        dg = pg.GInterpModalSerendipity(data, options.modalSerendipity)
        coords, values = dg.project(options.component)
        numDims = data.numDims
    elif options.maxOrder:
        dg = pg.GInterpMaxOrder(data, options.maxOrder)
        coords, values = dg.project(options.component)
        numDims = data.numDims
    else:
        c = [centeredLinspace(data.lowerBounds[d],
                              data.upperBounds[d],
                              data.numCells[d])
             for d in range(data.numDims)]
        coords = numpy.meshgrid(*c, indexing='ij')
        values = data.q[..., options.component]
        numDims = data.numDims
elif options.fNameRoot:
    hist = pg.GHistoryData(options.fNameRoot)
    coords = numpy.expand_dims(hist.time, axis=0)
    values = hist.values
    numDims = 1
else:
    print('Nothing specified to plot')
    sys.exit()

# plotting parameters are based solely on the personal taste of Ammar :)
plt.rcParams['lines.linewidth']            = 2
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
plt.rcParams['image.interpolation']        = 'none'
plt.rcParams['image.origin']               = 'lower'
plt.rcParams['contour.negative_linestyle'] = 'solid'
#plt.rcParams['savefig.bbox']               = 'tight'
#plt.rcParams['mathtext.default']           = 'regular'

# this needs to be last
if options.xkcd:
    plt.xkcd()

# plotting 
fig, ax = plt.subplots()
if numDims == 1:
    im = ax.plot(coords[0], values,
                 color=options.color)
elif numDims == 2:
    if not options.contour:
        im = ax.pcolormesh(coords[0], coords[1], values)
    else:
        im = ax.contour(coords[0], coords[1], values)
else:
    raise exceptions.RuntimeError(
        "Plotting 3D data is not currently supported")

# formating
ax.set_xlabel(options.xlabel)
ax.set_ylabel(options.ylabel)
ax.grid(options.grid)
if numDims == 1:
    plt.autoscale(enable=True, axis='x', tight=True)
elif numDims == 2:
    fig.colorbar(im)
    if options.eqvAxis:
        ax.axis('tight')
    else:
        ax.axis('image')

plt.tight_layout()

# this should be last
if options.xkcd:
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])

if not options.dontShow:
    plt.show()
