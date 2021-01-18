import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import click

import postgkyl.output.plot as gplot
from postgkyl.data import Data
import postgkyl.data.select as select
from postgkyl.commands.util import vlog, pushChain

def update(i, ctx, tag, kwargs):
    # if kwargs['collected']:
    #     grid, vals = select(ctx.obj['dataSets'][ctx.obj['sets'][0]],z0=i)
    #     dat = Data()
    #     dat.push(vals, grid=grid)
    #     dat.frame = i
    #     dat.time = grid[0][0]
    # else:
    #     dat = ctx.obj['dataSets'][ctx.obj['sets'][i]]
    dat = ctx.obj['data'].getDataset(tag, i)
    plt.clf()
    kwargs['title'] = ''
    if dat.frame is not None:
        kwargs['title'] = kwargs['title'] + 'F: {:d} '.format(dat.frame)
    #end
    if dat.time is not None:
        kwargs['title'] = kwargs['title'] + 'T: {:.4e}'.format(dat.time)
    #end
    if kwargs['arg'] is not None:
        return gplot(dat, kwargs['arg'], **kwargs)
    else:
        return gplot(dat, **kwargs)
    #end
#end

@click.command()
@click.option('--tag', '-t', default=None,
              help="Specify a tag to plot.")
@click.option('--squeeze', '-s', is_flag=True,
              help="Squeeze the components into one panel.")
@click.option('--subplots', '-b', is_flag=True,
              help="Make subplots from multiple datasets.")
@click.option('--nsubplotrow', 'nSubplotRow', type=click.INT,
              help="Manually set the number of rows for subplots.")
@click.option('--nsubplotcol', 'nSubplotCol', type=click.INT,
              help="Manually set the number of columns for subplots.")
@click.option('--transpose', is_flag=True,
              help="Transpose axes.")
@click.option('-c', '--contour', is_flag=True,
              help="Make contour plot.")
@click.option('-q', '--quiver', is_flag=True,
              help="Make quiver plot.")
@click.option('-l', '--streamline', is_flag=True,
              help="Make streamline plot.")
@click.option('-g', '--group', type=click.Choice(['0', '1']),
              help="Switch to group mode.")
@click.option('-s', '--scatter', is_flag=True,
              help="Make scatter plot.")
@click.option('--markersize', type=click.FLOAT,
              help="Set marker size for scatter plots.")
@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('-d', '--diverging', is_flag=True,
              help="Switch to diverging colormesh mode.")
@click.option('--arg', type=click.STRING,
              help="Additional plotting arguments, e.g., '*--'.")
@click.option('-a', '--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
@click.option('--logx', is_flag=True,
              help="Set x-axis to log scale.")
@click.option('--logy', is_flag=True,
              help="Set y-axis to log scale.")
@click.option('--logz', is_flag=True,
              help="Set values of 2D plot to log scale.")
@click.option('--xscale', default=1.0, type=click.FLOAT,
              help="Value to scale the x-axis (default: 1.0).")
@click.option('--yscale', default=1.0, type=click.FLOAT,
              help="Value to scale the y-axis (default: 1.0).")
@click.option('--vmax', default=None, type=click.FLOAT,
              help="Set maximal value of data for plots.")
@click.option('--vmin', default=None, type=click.FLOAT,
              help="Set minimal value of data for plots.")
@click.option('-f', '--float', is_flag=True,
              help="Choose min/max levels based on current frame (i.e., each frame uses a different color range).")
@click.option('--xlim', default=None, type=click.STRING,
              help="Set limits for the x-coordinate (lower,upper)")
@click.option('--ylim', default=None, type=click.STRING,
              help="Set limits for the y-coordinate (lower,upper).")
@click.option('--legend/--no-legend', default=True,
              help="Show legend.")
@click.option('--force-legend', 'forcelegend', is_flag=True,
              help="Force legend even when plotting a single dataset.")
@click.option('-x', '--xlabel', type=click.STRING,
              help="Specify a x-axis label.")
@click.option('-y', '--ylabel', type=click.STRING,
              help="Specify a y-axis label.")
@click.option('--clabel', type=click.STRING,
              help="Specify a label for colorbar.")
@click.option('--title', type=click.STRING,
              help="Specify a title.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
@click.option('--fps', type=click.INT,
              help="Specify frames per second for saving.")
@click.option('--dpi', type=click.INT,
              help="DPI (resolution) for output.")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges (default: None)")
@click.option('--showgrid/--no-showgrid', default=True,
              help="Show grid-lines (default: True)")
@click.option('--collected', is_flag=True,
              help="Animate a dataset that has been collected, i.e. a single dataset with time taken to be the first index.")
@click.option('--hashtag', is_flag=True,
              help="Turns on the pgkyl hashtag!")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
@click.pass_context
def animate(ctx, **kwargs):
    r"""Animate the actively loaded dataset and show resulting plots in a
    loop. Typically, the datasets are loaded using wildcard/regex
    feature of the -f option to the main pgkyl executable. To save the
    animation ffmpeg needs to be installed.

    """
    vlog(ctx, 'Starting animate')
    pushChain(ctx, 'animate', **kwargs)

    if not kwargs['float']:
        vmin = float('inf')
        vmax = float('-inf')
        for dat in ctx.obj['data'].iterator(kwargs['tag']):
            val = dat.getValues()
            if kwargs['logz']:
                val = np.log(val)
            #end
            if vmin > np.nanmin(val):
                vmin = np.nanmin(val)
            #end
            if vmax < np.nanmax(val):
                vmax = np.nanmax(val)
            #end
        if kwargs['vmin'] is None:
            kwargs['vmin'] = vmin
            if kwargs['logz']:
                kwargs['vmin'] = np.exp(vmin)
            #end
        #end
        if kwargs['vmax'] is None:
            kwargs['vmax'] = vmax
            if kwargs['logz']:
                kwargs['vmax'] = np.exp(vmax)
            #end
        #end
    #end

    # if kwargs['collected']:
    #     numSets = len(ctx.obj['dataSets'][ctx.obj['sets'][0]].getGrid()[0])
    # else:
    #     numSets = len(ctx.obj['sets'])
    numSets = ctx.obj['data'].getNumDatasets(kwargs['tag'])

    fig = plt.figure()
    kwargs['figure'] = fig
    kwargs['legend'] = False

    anim = []
    if kwargs['tag']:
        anim.append(FuncAnimation(fig, update, numSets,
                                  fargs=(ctx, kwargs['tag'], kwargs),
                                  interval=kwargs['interval'], blit=False))
    else:
        for t in ctx.obj['data'].tagIterator():
            anim.append(FuncAnimation(fig, update, numSets,
                                      fargs=(ctx, t, kwargs),
                                      interval=kwargs['interval'], blit=False))
        #end
    #end

    fName = 'anim.mp4'
    if kwargs['saveas']:
        fName = str(kwargs['saveas'])
    #end
    if kwargs['save'] or kwargs['saveas']:
        anim.save(fName, writer='ffmpeg',
                  fps=kwargs['fps'], dpi=kwargs['dpi'])
    #end
    
    if kwargs['show']:
        plt.show()
    #end
    vlog(ctx, 'Finishing animate')
#end
