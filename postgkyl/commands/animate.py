import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import click

import postgkyl.output.plot as gplot
from postgkyl.commands.util import vlog, pushChain

def update(s, ctx, kwargs):
    dat = ctx.obj['dataSets'][s]
    plt.clf()
    kwargs['title'] = 'F: {:d}  T: {:.4e}'.format(dat.frame, dat.time)
    if kwargs['arg'] is not None:
        return gplot(dat, kwargs['arg'], **kwargs)
    else:
        return gplot(dat, **kwargs)

@click.command(help='Animate the data')
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
# @click.option('--legend/--no-legend', default=True,
#               help="Show legend.")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
#@click.option('--color', type=click.STRING,
#              help="Set color when available.")
@click.option('-x', '--xlabel', type=click.STRING,
              help="Specify a x-axis label.")
@click.option('-y', '--ylabel', type=click.STRING,
              help="Specify a y-axis label.")
@click.option('-t', '--title', type=click.STRING,
              help="Specify a title label.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
@click.option('-f', '--float', is_flag=True,
              help="Choose min/max levels based on current frame")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges (default: None)")
@click.pass_context
def animate(ctx, **kwargs):
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
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
    #end

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
        anim.save(fName, writer='ffmpeg')
    #end
    
    if kwargs['show']:
        plt.show()
    #end
    vlog(ctx, 'Finishing animate')
#end
