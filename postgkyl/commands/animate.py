import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import click

import postgkyl.output.plot as gplot
from postgkyl.data import Data
import postgkyl.data.select as select
from postgkyl.commands.util import vlog, pushChain

def update(i, ctx, kwargs):
    if kwargs['collected']:
        grid, vals = select(ctx.obj['dataSets'][ctx.obj['sets'][0]],z0=i)
        dat = Data()
        dat.push(vals, grid=grid)
        dat.frame = i
        dat.time = grid[0][0]
    else:
        dat = ctx.obj['dataSets'][ctx.obj['sets'][i]]
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
@click.option('--squeeze', '-s', is_flag=True,
              help="Squeeze the components into one panel.")
@click.option('-a', '--arg', type=click.STRING,
              help="Additional plotting arguments like '*--'.")
@click.option('-c', '--contour', is_flag=True,
              help="Switch to contour mode.")
@click.option('-q', '--quiver', is_flag=True,
              help="Switch to quiver mode.")
@click.option('-l', '--streamline', is_flag=True,
              help="Switch to streamline mode.")
@click.option('-d', '--diverging', is_flag=True,
              help="Switch to diverging colormesh mode.")
@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
@click.option('--logx', is_flag=True,
              help="Set x-axis to log scale.")
@click.option('--logy', is_flag=True,
              help="Set y-axis to log scale.")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
@click.option('--color', type=click.STRING,
              help="Set color when available.")
@click.option('-x', '--xlabel', type=click.STRING,
              help="Specify a x-axis label.")
@click.option('-y', '--ylabel', type=click.STRING,
              help="Specify a y-axis label.")
@click.option('-t', '--title', type=click.STRING,
              help="Specify a title label.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
@click.option('-f', '--float', is_flag=True,
              help="Choose min/max levels based on current frame (i.e. each frame uses a different color range).")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
@click.option('--fps', type=click.INT,
              help="Specify frames per second for saving.")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges (default: None)")
@click.option('--vmax', default=None, type=click.FLOAT,
              help="Set maximal value of data for plots.")
@click.option('--vmin', default=None, type=click.FLOAT,
              help="Set minimal value of data for plots.")
@click.option('--collected', is_flag=True,
              help="Animate a dataset that has been collected, i.e. a single dataset with time taken to be the first index.")
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
        for s in ctx.obj['sets']:
            val = ctx.obj['dataSets'][s].getValues()
            if vmin > val.min():
                vmin = val.min()
            #end
            if vmax < val.max():
                vmax = val.max()
            #end
        if kwargs['vmin'] is None:
            kwargs['vmin'] = vmin
        #end
        if kwargs['vmax'] is None:
            kwargs['vmax'] = vmax
        #end
    #end

    if kwargs['collected']:
        numSets = len(ctx.obj['dataSets'][ctx.obj['sets'][0]].getGrid()[0])
    else:
        numSets = len(ctx.obj['sets'])
    fig = plt.figure()
    kwargs['figure'] = fig
    kwargs['legend'] = False
   
    anim = FuncAnimation(fig, update, numSets,
                         fargs=(ctx, kwargs),
                         interval=kwargs['interval'], blit=False)

    fName = 'anim.mp4'
    if kwargs['saveas']:
        fName = str(kwargs['saveas'])
    #end
    if kwargs['save'] or kwargs['saveas']:
        anim.save(fName, writer='ffmpeg', fps=kwargs['fps'])
    #end
    
    if kwargs['show']:
        plt.show()
    #end
    vlog(ctx, 'Finishing animate')
#end
