import matplotlib.pyplot as plt

import click

import postgkyl.output.plot as gplot
from postgkyl.commands.util import vlog, pushChain

@click.command()
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
@click.option('-d', '--diverging', is_flag=True,
              help="Switch to diverging colormesh mode.")
@click.option('-g', '--group', type=click.Choice(['0', '1']),
              help="Switch to group mode.")
@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
@click.option('--logx', is_flag=True,
              help="Set x-axis to log scale.")
@click.option('--logy', is_flag=True,
              help="Set y-axis to log scale.")
@click.option('--logz', is_flag=True,
              help="Set values of 2D plot to log scale.")
@click.option('--xscale', default=1.0, type=click.FLOAT,
              help="Scalar value to scale the x-axis (default: 1.0).")
@click.option('--yscale', default=1.0, type=click.FLOAT,
              help="Scalar value to scale the y-axis (default: 1.0).")
@click.option('--vmax', default=None, type=click.FLOAT,
              help="Set maximal value for plots.")
@click.option('--vmin', default=None, type=click.FLOAT,
              help="Set minimal value for plots.")
@click.option('--legend/--no-legend', default=True,
              help="Show legend.")
@click.option('--force-legend', 'forcelegend', is_flag=True,
              help="Force legend even when plotting a single dataset.")
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
@click.option('--dpi', type=click.INT, default=200,
              help="DPI for output")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges (default: None)")
@click.pass_context
def plot(ctx, **kwargs):
    vlog(ctx, 'Starting plot')
    pushChain(ctx, 'plot', **kwargs)

    if kwargs['group'] is not None:
        kwargs['group'] = int(kwargs['group'])
    #end

    for s in ctx.obj['sets']:
        dat = ctx.obj['dataSets'][s]
        if len(ctx.obj['sets']) > 1 or kwargs['forcelegend']:
            label = ctx.obj['labels'][s]
        else:
            label = ''
        #end
        if kwargs['arg'] is not None:
            gplot(dat, kwargs['arg'], labelPrefix=label, 
                 **kwargs)
        else:
            gplot(dat, labelPrefix=label,
                 **kwargs)
        #end

        if (kwargs['save'] or kwargs['saveas']):
            if kwargs['saveas']:
                fName = kwargs['saveas']
            else:
                if dat.fName:
                    s = dat.fName.split('.')
                else:
                    s = 'ev_' + ctx.obj['labels'][s].replace(' ', '_')
                #end
                fName = s + '.png'
            #end
        #end
        if (kwargs['save'] or kwargs['saveas']) and kwargs['figure'] is None:
            plt.savefig(fName, dpi=kwargs['dpi'])
        #end
    #end
    if (kwargs['save'] or kwargs['saveas']) and kwargs['figure'] is not None:
        plt.savefig(fName, dpi=kwargs['dpi'])
    #end

    if kwargs['show']:
        plt.show()
    #end
    vlog(ctx, 'Finishing plot')
#end
