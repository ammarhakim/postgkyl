import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import tables
from time import time

from postgkyl.tools.stack import peakStack, peakLabel, getFullLabel

dirPath = os.path.dirname(os.path.realpath(__file__))

#---------------------------------------------------------------------
#-- Plotting ---------------------------------------------------------
def _colorbar(obj, _ax, _fig, redraw=False, aspect=None, label=''):
    """Add a colorbar adjacent to obj, with a matching height

    For use of aspect, see:
    http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_aspect
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _divider_ = make_axes_locatable(_ax)
    _cax_ = _divider_.append_axes("right", size="5%", pad=0.05)
    _cbar_ = _fig.colorbar(obj, cax=_cax_, label=label)
    if aspect is not None:
        _ax.set_aspect(aspect)
    if redraw:
        _fig.canvas.draw()
    return _cbar_

def _getFig(ctx):
    if ctx.obj['hold'] == 'on' and ctx.obj['fig'] != '':
        fig = ctx.obj['fig']
        ax = ctx.obj['ax']
    else:
        fig, ax = plt.subplots()
        ctx.obj['fig'] = fig
        ctx.obj['ax'] = ax
    return fig, ax

@click.command(help='Plot the data')
@click.option('--show/--no-show', default=True,
              help='Turn showing of the plot ON and OFF (default: ON)')
@click.option('--style',
              help='Specify Matplotlib style file (default: postgkyl style)')
@click.option('--fixed-axis', 'axismode', flag_value='image')
@click.option('--free-axis', 'axismode', flag_value='tight',
              default=True)
@click.option('--save/--no-save', '-s', default=False,
              help='Save figure as png')
@click.option('-q', '--quiver', is_flag=True,
              help='Switch to quiver mode')
@click.option('-l', '--streamline', is_flag=True,
              help='Switch to streamline mode')
@click.option('-c', '--contour', is_flag=True,
              help='Switch to contour mode')
@click.option('--color', type=click.STRING,
              help='Set color for some plots')
@click.option('--logx', is_flag=True,
              help='Set x-axis to log scale')
@click.option('--logy', is_flag=True,
              help='Set y-axis to log scale')
@click.option('--legend/--no-legend', default=True,
              help='Show legend')
@click.pass_context
def plot(ctx, show, style, axismode, save,
         quiver, contour, streamline,
         color, logx, logy, legend):
    vlog(ctx, 'Starting plot')
    if style is None:
        plt.style.use(ctx.obj['mplstyle'])
    else:
        plt.style.use(style)

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        label = peakLabel(ctx, s)
        title = peakLabel(ctx, s, 0)

        numDims = len(coords)
        # Sometimes an array will have only one cell in some dimension,
        # get rid of them
        idx = []
        for d in range(numDims):
            if len(coords[d]) == 1:
                idx.append(d)
        if len(idx) > 0:
            coords = np.delete(coords, idx)
            values = np.squeeze(values, tuple(idx))

        numDims = len(coords)
        numComps = values.shape[-1]
        if streamline or quiver:
            skip = 2
        else:
            skip = 1
        idxComps = range(0, numComps, skip)
        for comp in idxComps:
            vlog(ctx, 'Plotting set #{:d} component #{:d}'.format(s, comp))  
            fig, ax = _getFig(ctx)

            # Specialized plotting
            if contour:
                im = ax.contour(coords[0], coords[1],
                                values[..., comp].transpose())
                cb =  _colorbar(im, ax, fig)
            elif quiver:
                skip = int(np.max((len(coords[0]), len(coords[1])))//15)
                skip2 = int(skip//2)
                im = ax.quiver(coords[0][skip2::skip], coords[1][skip2::skip],
                               values[skip2::skip,
                                      skip2::skip,
                                      comp].transpose(),
                               values[skip2::skip,
                                      skip2::skip,
                                      comp+1].transpose())
            elif streamline:
                magnitude = np.sqrt(values[..., comp]**2 + 
                                    values[..., comp+1]**2)
                im = ax.streamplot(coords[0], coords[1],
                                   values[..., comp].transpose(),
                                   values[..., comp+1].transpose(),
                                   color=magnitude.transpose())
                cb = _colorbar(im.lines, ax, fig)
            # Default plotting
            else:
                if numDims == 1:
                    im, = ax.plot(coords[0], values[..., comp])
                elif numDims == 2:
                    im = ax.pcolormesh(coords[0], coords[1],
                                       values[..., comp].transpose())
                    cb = _colorbar(im, ax, fig)
                else:
                    click.echo('plot: {:d}D plots currently not supported'.
                               format(numDims))
                    ctx.exit()

            # Formating
            if len(idxComps) > 1:
                labelComp = '{:s} c{:d}'.format(label, comp)
            else:
                labelComp = label

            if color is not None:
                if color == 'seq':
                    cl = cm.inferno(s/len(ctx.obj['sets']))
                else:
                    cl = color
                try:
                    im.set_color(cl)
                except:
                    im.lines.set_color(cl)
                    im.arrows.set_color(cl)

            if logx:
                ax.set_xscale('log')
            if logy:
                ax.set_yscale('log')

            try:
                im.set_label(labelComp)
            except:
                im.lines.set_label(labelComp)

            if ctx.obj['hold'] == 'on':
                ax.set_title('{:s}'.format(title), y=1.08)
                if legend:
                    ax.legend(loc=0)
            else:
                ax.set_title('{:s} {:s}'.format(title, labelComp), y=1.08)
                
            if numDims == 1:
                plt.autoscale(enable=True, axis='x', tight=True)
            elif numDims == 2:
                ax.axis(axismode)
                ax.set_xlim((coords[0][0], coords[0][-1]))
                ax.set_ylim((coords[1][0], coords[1][-1]))
            ax.grid(True)
            plt.tight_layout()

            if numComps > 1 and ctx.obj['hold'] == 'off':
                saveName = '{:s}_c{:d}.png'.format(getFullLabel(ctx, s),
                                                   comp)
            else:
                saveName = '{:s}.png'.format(getFullLabel(ctx, s))
            if save and ctx.obj['hold'] == 'off':
                fig.savefig(saveName, dpi=150)

    if save and ctx.obj['hold'] == 'on':
        vlog(ctx, 'plot: Saving figure as {:s}'.format(saveName))
        fig.savefig(saveName)
    if show:
        vlog(ctx, 'plot: Showing figure')
        plt.show()
    vlog(ctx, 'Finishing plot')

@click.command(help='Hold the plotting')
@click.option('--on', 'hld', flag_value='on', default=True,
              help='Turn plot hold ON')
@click.option('--off', 'hld', flag_value='off',
              help='Turn plot hold OFF')
@click.pass_context
def hold(ctx, hld):
    vlog(ctx, 'Hold set to {}'.format(hld))
    ctx.obj['hold'] = hld


#---------------------------------------------------------------------
#-- Info -------------------------------------------------------------
@click.command(help='Print the current top of stack info')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def info(ctx, allsets):
    vlog(ctx, 'Starting info')
    if allsets is True:
        click.echo('\nPrinting the current top of stack information (all data sets):')
        sets = range(ctx.obj['numSets'])
    else:
        click.echo('\nPrinting the current top of stack information (active data sets):')
        sets = ctx.obj['sets']
        
    for s in sets:
        coords, values = peakStack(ctx, s)
        click.echo(' * Dataset #{:d}'.format(ctx.obj['setIds'][s]))
        if ctx.obj['type'][s] == 'frame':
            click.echo('   * Time: {:e}'.format(ctx.obj['data'][s].time))
#        else:
#            click.echo('   * Time: {:e} - {:e}'.format(ctx.obj['data'][s].time[0],
#                                                      ctx.obj['data'][s].time[-1]))
        click.echo('   * Number of components: {:d}'.format(values.shape[-1]))
        click.echo('   * Minimum: {:e}'.format(values.min()))
        amin = np.unravel_index(np.argmin(values), values.shape)
        click.echo('     * Minimum Index: {:s}'.format(str(amin)))
        click.echo('   * Maximum: {:e}'.format(values.max()))
        amax = np.unravel_index(np.argmax(values), values.shape)
        click.echo('     * Maximum Index: {:s}'.format(str(amax)))
        numDims = len(coords)
        click.echo('   * Dimensions ({:d}):'.format(numDims))
        for d in range(numDims):
            if ctx.obj['type'][s] == 'frame':
                if len(coords[d]) > 1:
                    dx2 = 0.5*(coords[d][1] - coords[d][0])
                else:
                    dx2 = 0.0
                minC = coords[d][0]-dx2
                maxC = coords[d][-1]+dx2
            else:
                minC = coords[d][0]
                maxC = coords[d][-1]
            click.echo('     * Dim {:d}: Num. Cells: {:d}; Lower: {:e}; Upper: {:e}'.format(d, len(coords[d]), minC, maxC))
    vlog(ctx, 'Finishing info')


#---------------------------------------------------------------------
#-- Writing ----------------------------------------------------------
def flatten(coords, values):
    numDims = int(len(coords))
    numComps = int(values.shape[-1]) 
    numRows = int(len(values.flatten())/numComps)
    dataOut = np.zeros((numRows, numDims+numComps))

    numCells = np.zeros(numDims)
    for d in range(numDims):
        numCells[d] = values.shape[d]
    basis = np.full(numDims, 1.0)
    for d in range(numDims-1):
        basis[d] = numCells[(d+1):].prod()

    for i in range(numRows):
        idx = i
        idxs = []
        for d in range(numDims):
            idxs.append(int(idx // basis[d]))
            idx = idx % basis[d]
            
        for d in range(numDims):
            dataOut[i, d] = coords[d][idxs[d]]
        for c in range(numComps):
            dataOut[i, numDims+c] = values[tuple(idxs)][c]

    return dataOut

@click.command(help='Save the current top of stack into ASCII or H5 file')
@click.option('--filename', '-f', type=click.STRING)
@click.option('--mode', '-m', type=click.Choice(['h5', 'txt']),
              default='h5')
@click.pass_context
def write(ctx, filename, mode):
    vlog(ctx, 'Starting write in {:s} mode'.format(mode))
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        numDims = int(len(coords))
        numComps = int(values.shape[-1])

        if filename is None:
            filename = '{:s}.{:s}'.format(getFullLabel(ctx, s), mode)

        if mode == 'h5':
            fh = tables.open_file(filename, 'w')

            lowerBounds = np.zeros(numDims)
            upperBounds = np.zeros(numDims)
            numCells = np.zeros(numDims)
            for d in range(numDims):
                if ctx.obj['type'][s] == 'frame':
                    dx2 = 0.5*(coords[d][1] - coords[d][0])
                    lowerBounds[d] = coords[d][0]-dx2
                    upperBounds[d] = coords[d][-1]+dx2
                else:
                    lowerBounds[d] = coords[d][0]
                    upperBounds[d] = coords[d][-1]
                numCells[d] = values.shape[d]
            grid = fh.create_group('/', 'StructGrid')
            grid._v_attrs.vsLowerBounds = lowerBounds
            grid._v_attrs.vsUpperBounds = upperBounds
            grid._v_attrs.vsNumCells = numCells

            timeData = fh.create_group('/', 'timeData')
            if ctx.obj['type'][s] == 'frame':
                timeData._v_attrs.vsTime = ctx.obj['data'][s].time
            else:
                timeData._v_attrs.vsTime = -1.0
            fh.create_array('/', 'StructGridField', values)

            fh.close()

        elif mode == 'txt':
            np.savetxt(filename, flatten(coords, values))
    vlog(ctx, 'Finishing write')

def vlog(ctx, message):
    if ctx.obj['verbose']:
        elapsedTime = time() - ctx.obj['startTime']
        click.echo(click.style('[{:f}] {:s}'.format(elapsedTime, message), fg='green'))
    
        
            
