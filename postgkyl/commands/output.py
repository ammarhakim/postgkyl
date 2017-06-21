import click
import numpy as np
import matplotlib.pyplot as plt
import os
import tables

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
@click.option('--style', default=dirPath+'/postgkyl.mplstyle',
              help='Specify Matplotlib style file (default: Postgkyl style)')
@click.option('--fixed-axis', 'axismode', flag_value='image',
             default=True)
@click.option('--free-axis', 'axismode', flag_value='tight')
@click.option('--save/--no-save', '-s', default=False,
              help='Save figure as png')
@click.pass_context
def plot(ctx, show, style, axismode, save):
    if not os.path.isfile(style): # conda distribution path
        style = dirPath + '/../../../../../data/postgkyl.mplstyle'
    plt.style.use(style)

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        label = peakLabel(ctx, s)
        title = peakLabel(ctx, s, 0)
        if label == title:
            label = ''

        numDims = len(coords)
        numComps = values.shape[-1]

        for comp in range(numComps):
            fig, ax = _getFig(ctx)

            if numComps > 1:
                labelComp = '{:s} c{:d}'.format(label, comp)
            else:
                labelComp = label

            if numDims == 1:
                im = ax.plot(coords[0], values[..., comp],
                             label=labelComp)
                plt.autoscale(enable=True, axis='x', tight=True)
            elif numDims == 2:
                im = ax.pcolormesh(coords[0], coords[1],
                                   values[..., comp].transpose(),
                                   label=labelComp)
                _colorbar(im, ax, fig)
                ax.axis(axismode)
            else:
                click.echo('{:d}D plots currently not supported'.
                           format(numDims))
                ctx.exit()

            if ctx.obj['hold'] == 'on':
                ax.set_title('{:s}'.format(title))
                ax.legend(loc=0)
            else:
                ax.set_title('{:s} {:s}'.format(title, labelComp))

            ax.grid(True)
            plt.tight_layout()

            if numComps > 1 and ctx.obj['hold'] == 'off':
                saveName = '{:s}_c{:d}.png'.format(getFullLabel(ctx, s),
                                                   comp)
            else:
                saveName = '{:s}.png'.format(getFullLabel(ctx, s))
            if save and ctx.obj['hold'] == 'off':
                fig.savefig(saveName)

    if save and ctx.obj['hold'] == 'on':
        fig.savefig(saveName)
    if show:
        plt.show()

@click.command(help='Hold the plotting')
@click.option('--on', 'hld', flag_value='on', default=True,
              help='Turn plot hold ON')
@click.option('--off', 'hld', flag_value='off',
              help='Turn plot holf OFF')
@click.pass_context
def hold(ctx, hld):
    ctx.obj['hold'] = hld


#---------------------------------------------------------------------
#-- Info -------------------------------------------------------------
@click.command(help='Print the current top of stack info')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def info(ctx, allsets): 
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
            click.echo('   * Time: {:f}'.format(ctx.obj['data'][s].time))
        else:
            click.echo('   * Time: {:f} - {:f}'.format(ctx.obj['data'][s].time[0],
                                                      ctx.obj['data'][s].time[-1]))
        click.echo('   * Number of components: {:d}'.format(values.shape[-1]))
        click.echo('   * Minimum: {:f}'.format(values.min()))
        amin = np.unravel_index(np.argmin(values), values.shape)
        click.echo('     * Minimum Index: {:s}'.format(str(amin)))
        click.echo('   * Maximum: {:f}'.format(values.max()))
        amax = np.unravel_index(np.argmax(values), values.shape)
        click.echo('     * Maximum Index: {:s}'.format(str(amax)))
        numDims = len(coords)
        click.echo('   * Dimensions ({:d}):'.format(numDims))
        for d in range(numDims):
            if ctx.obj['type'][s] == 'frame':
                dx2 = 0.5*(coords[d][1] - coords[d][0])
                minC = coords[d][0]-dx2
                maxC = coords[d][-1]+dx2
            else:
                minC = coords[d][0]
                maxC = coords[d][-1]
            click.echo('     * Dim {:d}: Num. Cells: {:d}; Lower: {:f}; Upper: {:f}'.format(d+1, len(coords[d]), minC, maxC))


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

    idxs = np.zeros(numDims)
    for i in range(numRows):
        idx = i
        for d in range(numDims):
            idxs[d] = int(idx // basis[d])
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
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, dataset)

        numDims = int(len(coords))
        numComps = int(values.shape[-1])

        if filename is None:
            filename = '{:s}.{:s}'.format(getFullLabel(ctx, dataset), mode)

        if mode == 'h5':
            fh = tables.open_file(filename, 'w')

            lowerBounds = np.zeros(numDims)
            upperBounds = np.zeros(numDims)
            numCells = np.zeros(numDims)
            for d in range(numDims):
                lowerBounds[d] = coords[d].min()
                upperBounds[d] = coords[d].max()
                numCells[d] = values.shape[d]
            grid = fh.create_group('/', 'StructGrid')
            grid._v_attrs.vsLowerBounds = lowerBounds
            grid._v_attrs.vsUpperBounds = upperBounds
            grid._v_attrs.vsNumCells = numCells

            timeData = fh.create_group('/', 'timeData')
            timeData._v_attrs.vsTime = ctx.obj['data'][dataset].time

            fh.create_array('/', 'StructGridField', values)

            fh.close()

        elif mode == 'txt':
            np.savetxt(filename, flatten(coords, values))
        
            
