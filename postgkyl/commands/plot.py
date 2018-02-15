import matplotlib.pyplot as plt

import click

import postgkyl.output
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Plot the data')
@click.option('--figure', '-f', default=None,
              help="Specify figure to plot in (default: data set number)")
@click.option('--squeeze', '-s', is_flag=True,
              help="Specify is components should be squeezed into one Axes")
@click.option('--show/--no-show', default=True,
              help='Turn showing of the plot ON and OFF (default: ON)')
@click.option('--style',
              help='Specify Matplotlib style file (default: postgkyl style)')
@click.option('--free-axis', 'axismode', flag_value='tight', default=True)
@click.option('--fixed-axis', 'axismode', flag_value='image')
@click.option('--save/--no-save', default=False,
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
@click.option('--saveas', type=click.STRING, default=None,
              help='Name of PNG file to save')
@click.pass_context
def plot(ctx, **kwargs):
    vlog(ctx, 'Starting plot')
    pushChain(ctx, 'plot.plot', **kwargs)

    for s in ctx.obj['sets']:
        dat = ctx.obj['dataSets'][s]
        postgkyl.output.plot(dat, **kwargs)
        plt.show()

    vlog(ctx, 'Finishing plot')
