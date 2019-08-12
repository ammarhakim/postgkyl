import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import click

import postgkyl.output.plot as gplot
from postgkyl.commands.util import vlog, pushChain

def update(i, ax, dat):
    time = dat.getGrid()[0]
    coords = dat.getValues()
    plt.cla()
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2])
    plt.title('T: {:.4e}'.format(time[i]))
    ax.set_xlabel('$z_0$')
    ax.set_ylabel('$z_1$')
    ax.set_zlabel('$z_2$')
#end

@click.command(help='Animate a particle trajectory')
# @click.option('--squeeze', '-s', is_flag=True,
#               help="Squeeze the components into one panel.")
# @click.option('-a', '--arg', type=click.STRING,
#               help="Additional plotting arguments like '*--'.")
# @click.option('-c', '--contour', is_flag=True,
#               help="Switch to contour mode.")
# @click.option('-q', '--quiver', is_flag=True,
#               help="Switch to quiver mode.")
# @click.option('-l', '--streamline', is_flag=True,
#               help="Switch to streamline mode.")
# @click.option('-d', '--diverging', is_flag=True,
#               help="Switch to diverging colormesh mode.")
# @click.option('--style',
#               help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
# @click.option('--logx', is_flag=True,
#               help="Set x-axis to log scale.")
# @click.option('--logy', is_flag=True,
#               help="Set y-axis to log scale.")
# @click.option('--legend/--no-legend', default=True,
#               help="Show legend.")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
# #@click.option('--color', type=click.STRING,
# #              help="Set color when available.")
# @click.option('-x', '--xlabel', type=click.STRING,
#               help="Specify a x-axis label.")
# @click.option('-y', '--ylabel', type=click.STRING,
#               help="Specify a y-axis label.")
# @click.option('-t', '--title', type=click.STRING,
#               help="Specify a title label.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
# @click.option('-f', '--float', is_flag=True,
#               help="Choose min/max levels based on current frame")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
# @click.option('-e', '--edgecolors', type=click.STRING,
#               help="Set color for cell edges (default: None)")
@click.option('-e', '--elevation', type=click.FLOAT,
              help="Set elevation")
@click.option('-a', '--azimuth', type=click.FLOAT,
              help="Set azimuth")
@click.pass_context
def trajectory(ctx, **kwargs):
    vlog(ctx, 'Starting trajectory')
    pushChain(ctx, 'trajectory', **kwargs)

    numSets = len(ctx.obj['sets'])
    fig = plt.figure()
    ax = Axes3D(fig)
    kwargs['figure'] = fig
    kwargs['legend'] = False

    dat = ctx.obj['dataSets'][ctx.obj['sets'][0]]
    numPos = dat.getNumCells()[0]
   
    anim = FuncAnimation(fig, update, numPos,
                         fargs=(ax, dat,),
                         interval=kwargs['interval'])

    ax.view_init(elev=kwargs['elevation'], azim=kwargs['azimuth'])

    if kwargs['fixaspect']:
        plt.setp(ax, aspect=1.0)

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
    vlog(ctx, 'Finishing trajectory')
#end
