#!/usr/bin/env python
"""
Postgkyl script to plot data directly from the terminal
"""
# standart imports
import numpy
import matplotlib.pyplot as plt
import sys
import os
from optparse import OptionParser
# custom imports
import postgkyl as pg

# --------------------------------------------------------------------
# Parser -------------------------------------------------------------
usage = "usage: %prog [options] file1 file2 ..."
parser = OptionParser(usage=usage)
# What to plot
parser.add_option('-c', '--component', action='append',
                  dest='component',
                  help='Component to plot (default 0; multiple -c calls are allowed)')
parser.add_option('-m', '--mask', action='store',
                  dest='maskName',
                  help='G file that serves as a mask')
parser.add_option('--surf3D', action='store',
                  dest='surf3D',
                  help='Select isosurface for 3D plotting')
# projecting
parser.add_option('--ns', action='store',
                  dest='nodalSerendipity',
                  help='Polynomial order of the nodal Serendipity basis')
parser.add_option('--ms', action='store',
                  dest='modalSerendipity',
                  help='Polynomial order of the modal Serendipity basis')
parser.add_option('--mo', action='store',
                  dest='maxOrder',
                  help='Polynomial order of the max order basis')
# saving plot
parser.add_option('-s', '--save', action='store_true',
                  dest='save',
                  help='Save the displayed plot (png by default)')
parser.add_option('--saveAs', action='store',
                  dest='saveAs',
                  help='When saving figures, use this file name')
# how to plot
parser.add_option('--style', action='store',
                  dest='style', default='',
                  help='Selects MPL style file to use')
parser.add_option('--xlabel', action='store',
                  dest='xlabel', default='',
                  help='Set x-label')
parser.add_option('--ylabel', action='store',
                  dest='ylabel', default='',
                  help='Set y-label')
parser.add_option('-t', '--title', action='store',
                  dest='title',
                  help='Set title')
parser.add_option('--no-title', action='store_false',
                  dest='titleOn', default=True,
                  help='Turn OFF the title')
parser.add_option('-g', '--no-grid', action='store_false',
                  dest='gridOn', default=True,
                  help='Turn OFF the grid')
parser.add_option('--cmap', action='store',
                  dest='cmap', default='inferno',
                  help='Color map to use for 2D plots (default \'inferno\')')
parser.add_option('--axis-free', action='store_true',
                  dest='freeAxis',
                  help="Turn OFF equal axis scalling",
                  default=False)
parser.add_option('--contour', action='store_true',
                  dest='contour', default=False,
                  help="Plot contour instead of a bitmat for 2D plots")
# misellaneous
parser.add_option('-i', '--info', action='store_true',
                  dest='info', default=False,
                  help='Print information about the file(s)')
parser.add_option('--dont-show', action='store_true',
                  dest='dontShow', default=False,
                  help='Do NOT show the figure')
parser.add_option('-x', '--xkcd', action='store_true',
                  dest='xkcd', default=False,
                  help='Plot xkcd.com style plots!')
parser.add_option('-w', '--write-history', action='store_true',
                  dest='writeHistory', default=False,
                  help='Write the loaded history to a text file')
# Fixing components (slices)
parser.add_option('--fix1', action='store',
                  dest='fix1', default=None,
                  help='Fix the first coordinate on set value')
parser.add_option('--fix2', action='store',
                  dest='fix2', default=None,
                  help='Fix the second coordinate on set value')
parser.add_option('--fix3', action='store',
                  dest='fix3', default=None,
                  help='Fix the third coordinate on set value')
parser.add_option('--fix4', action='store',
                  dest='fix4', default=None,
                  help='Fix the fourth coordinate on set value')
parser.add_option('--fix5', action='store',
                  dest='fix5', default=None,
                  help='Fix the fifth coordinate on set value')
parser.add_option('--fix6', action='store',
                  dest='fix6', default=None,
                  help='Fix the sixth coordinate on set value')
(options, args) = parser.parse_args()

def _guessFileType(fName):
    ext = fName.split('.')[-1]
    if ext == 'bp' or ext == 'h5':
        return 'frame'
    else:
        return 'hist'

files = args
if files == []:
    print(' *** No data files specified. Exiting')
    sys.exit()
numFiles = len(files)
# determine the mode
mode = _guessFileType(files[0])

if options.component:
    components = options.component
else:
    components = [0]
numComps = len(components)
numData = numFiles*numComps

# --------------------------------------------------------------------
# Data Loading -------------------------------------------------------
def _loadFrame(fName, comp, numData, title):
    data = pg.GData(fName)
    if options.nodalSerendipity:
        dg = pg.GInterpNodalSerendipity(data, int(options.nodalSerendipity))
        coords, values = dg.project(int(comp))
    elif options.modalSerendipity:
        dg = pg.GInterpModalSerendipity(data, int(options.modalSerendipity))
        coords, values = dg.project(int(comp))
    elif options.maxOrder:
        dg = pg.GInterpModalMaxOrder(data, int(options.maxOrder))
        coords, values = dg.project(int(comp))
    else:  # fake interpolator for finite volume data
        dg = pg.data.GInterpZeroOrder(data)
        coords, values = dg.project(int(comp))

    # masking
    if options.maskName:
        maskField = pg.GData(options.maskName).q[..., 0]
        values = numpy.ma.masked_where(maskField < 0.0, values)
    # slicing
    coords, values = pg.tools.fixCoordSlice(coords, values, 'value',
                                            options.fix1, options.fix2,
                                            options.fix3, options.fix4,
                                            options.fix5, options.fix6)
    if numData == 1:
        title = title + '\nt: {:1.2e}'.format(data.time) 
    return coords, values, len(values.shape), title

def _loadHistory(fNameRoot, comp, numData, title):
    hist = pg.GHistoryData(fNameRoot)
    coords = numpy.expand_dims(hist.time, axis=0)
    values = hist.values
    if len(values.shape) > 1:
        values = values[:, int(comp)]
    values = numpy.squeeze(values)
    if numData == 1:
        title = title + '\nt: {:1.2e} .. {:1.2e}'.format(coords[0,0],
                                                         coords[0,-1]) 
    return coords, values, 1, title


# --------------------------------------------------------------------
# Info functions -----------------------------------------------------
def _printInfoFrame(fName):
    data = pg.GData(fName)
    print('Printing file info:')
    print(' * File name: {:s}'.format(data.fName))
    print(' * Time: {:f}'.format(data.time))
    print(' * Dimensions ({:d}):'.format(data.numDims))
    for i in range(data.numDims):
        print('   * Dim {:d}: Num. Cells: {:d}; Lower: {:f}; Upper: {:f}'.
              format(i+1, data.numCells[i],
                     data.lowerBounds[i], data.upperBounds[i]))


def _printInfoHistory(fNameRoot):
    hist = pg.GHistoryData(fNameRoot)
    print('Printing files info:')
    print(' * File names: {:s} .. {:s} ({:d} files)'.
          format(hist.files[0], hist.files[-1], len(hist.files)))
    print(' * Time: {:f} - {:f}'.format(hist.time[0], hist.time[-1]))


# --------------------------------------------------------------------
# Creating Titles and Names ------------------------------------------
if numData == 1:
    name = files[0]
    name = name.split('/')[-1]  # get rid of the full path

    if mode == 'frame':
        name = ''.join(name.split('.')[: -1])  # get rid of the extension
        # This weird Python construct is here in case someone would
        # like to use '.' in name... I really dislike it but I don't
        # know about any better -pc
    elif mode == 'hist':
        name = name.strip('0')
        name = name.strip('_')

    if options.title is None:
        if mode == 'frame':
            title = name
        elif mode == 'hist':
            title = name + '*'
        if options.component:
            title = '{} C:{:d}'.format(title, int(components[0]))
    else:
        title = str(options.title)

    if options.saveAs is None:
        if options.component:
            saveName = '{}_C{:d}'.format(name, int(components[0]))
            saveName = '{}/{}.png'.format(os.getcwd(), saveName)
    else:
        saveName = str(options.saveAs)
else:
    name = 'multi-plot'
    title = 'multi-plot'
    if options.saveAs is None:
        saveName = '{}/multi-plot.png'.format(os.getcwd())
    else:
        saveName = str(options.saveAs)

# --------------------------------------------------------------------
# Plotting setup -----------------------------------------------------
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['image.cmap'] = str(options.cmap)

# load personal Matplotlib style file
if options.style:
    plt.style.use(str(options.style))

# this needs to be set after the rest of rcParams
if options.xkcd:
    plt.rcParams['mathtext.default'] = 'regular'
    plt.xkcd()

# --------------------------------------------------------------------
# Plotting -----------------------------------------------------------
if not options.info:
    fig, ax = plt.subplots()

for i, fl in enumerate(files):
    for j, comp in enumerate(components):
        # first check if info option is on
        if options.info:
            if mode == 'frame':
                _printInfoFrame(fl)
            elif mode == 'hist':
                _printInfoHistory(fl)
            continue

        # loading
        if mode == 'frame':
            try:
                coords, values, numDims, title  = _loadFrame(fl,
                                                             int(comp),
                                                             numData,
                                                             title)
            except:
                if i == 0:  # allow mode switch only for the first file
                    coords, values, numDims, title = _loadHistory(fl,
                                                                  int(comp),
                                                                  numData,
                                                                  title)
                    mode = 'hist'
                else:
                    print(' *** Mixed \'frame\' and \'history\' data on input. Exiting')
                    sys.exit()
                
        elif mode == 'hist':
            try:
                coords, values, numDims, title = _loadHistory(fl,
                                                              int(comp),
                                                              numData,
                                                              title)
            except:
                if i == 0:  # allow mode switch only for the first file
                    coords, values, numDims, title  = _loadFrame(fl,
                                                                 int(comp),
                                                                 numData,
                                                                 title)
                    mode = 'frame'
                else:
                    print(' *** Mixed \'frame\' and \'history\' data on input. Exiting')
                    sys.exit()               

        # plotting
        if numDims == 1:
            if not options.xkcd:
                im = ax.plot(coords[0], values,
                             label='{:d}:{:d}'.format(i+1, j))
            else:
                im = ax.plot(coords[0], values,
                             label='{:d}:{:d}'.format(i+1, j),
                             clip_on=False, zorder=100)
        elif numDims == 2:
            if i > 0 or j > 0:
                print("Cannot plot more that one dataset for 2D. Skipping")
                continue
            if not options.contour:
                im = ax.pcolormesh(coords[0], coords[1], values.transpose())
            else:
                im = ax.contour(coords[0], coords[1], values.transpose())
        elif numDims == 3:
            if i > 0 or j > 0:
                print("Cannot plot more that one dataset for 3D. Skipping")
                continue
            if options.surf3D:
                from skimage import measure
                from mpl_toolkits.mplot3d import Axes3D
                plt.close(fig)

                verts, faces = measure.marching_cubes(values, float(options.surf3D))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2])
            else:
                raise RuntimeError(
                    "Isosurface value needs to be specified for 3D plotting.\nUse the flag --surf3D.")

        else:
            raise RuntimeError(
                "Plotting {}D plot? Seriously?".format(numDims))

if options.info:
    sys.exit()

# --------------------------------------------------------------------
# Formating ----------------------------------------------------------
def _colorbar(obj, _ax, _fig, redraw=False, aspect=None, label=''):
    """Add a colorbar adjacent to obj, with a matching height

    For use of aspect, see:
    http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_aspect
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _divider_ = make_axes_locatable(_ax)
    _cax_ = _divider_.append_axes("right", size="5%", pad=0.05)
    _cbar_ = fig.colorbar(obj, cax=_cax_, label=label)
    if aspect is not None:
        _ax.set_aspect(aspect)
    if redraw:
        _fig.canvas.draw()
    return _cbar_

if options.titleOn:
    if len(files) == 1 or options.title is not None:
        ax.set_title(title)
ax.set_xlabel(str(options.xlabel))
ax.set_ylabel(str(options.ylabel))
ax.grid(options.gridOn)
if numFiles > 1:
    ax.legend(loc=0)
if numDims == 1:
    plt.autoscale(enable=True, axis='x', tight=True)
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
    # Turn OFF the grid
    ax.grid(False)

if options.save or options.saveAs:
    fig.savefig(saveName, bbox_inches='tight', dpi=200)

if options.writeHistory:
    hist.save()

if not options.dontShow:
    plt.show()
else:
    plt.close(fig)
