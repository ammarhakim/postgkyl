import matplotlib.pyplot as plt

import click

import postgkyl.output
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Plot the data')
@click.option('--figure', '-f', default=None,
              help="Specify figure to plot in.")
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

@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('--fixed-axis', 'fixedaxis', is_flag=True,
              help="Enforce the same scaling on both axes.")
@click.option('--logx', is_flag=True,
              help="Set x-axis to log scale.")
@click.option('--logy', is_flag=True,
              help="Set y-axis to log scale.")
@click.option('--legend/--no-legend', default=True,
              help="Show legend.")
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
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
@click.pass_context
def plot(ctx, **kwargs):
    vlog(ctx, 'Starting plot')
    pushChain(ctx, 'plot.plot', **kwargs)

    for s in ctx.obj['sets']:
        dat = ctx.obj['dataSets'][s]
        #if kwargs['color'] == 'seq':
        #    kwargs['color'] = cm.inferno(s/len(ctx.obj['sets']))
        if kwargs['arg'] is not None:
            postgkyl.output.plot(dat, kwargs['arg'], labelPrefix='s{:d}'.format(s),
                                 **kwargs)
        else:
            postgkyl.output.plot(dat, labelPrefix='s{:d}'.format(s),
                                 **kwargs)
        if kwargs['save'] or kwargs['saveas']:
            if kwargs['saveas']:
                fName = kwargs['saveas']
            else:
                s = dat.fName.split('.')
                fName = s[0] + '_plot.png'
            plt.savefig(fName, dpi=200)

    if kwargs['show']:
        plt.show()
    vlog(ctx, 'Finishing plot')
