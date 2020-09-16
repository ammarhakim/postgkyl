import matplotlib.pyplot as plt

import click

import postgkyl.output.plot as gplot
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--figure', '-f', default=None,
              help="Specify figure (integer) to plot in.")
@click.option('--squeeze', '-s', is_flag=True,
              help="Squeeze the components into one panel.")
@click.option('--subplots', '-b', is_flag=True,
              help="Make subplots from multiple datasets.")
@click.option('--arg', type=click.STRING,
              help="Additional plotting arguments like '*--'.")
@click.option('-c', '--contour', is_flag=True,
              help="Draw contour plot.")
@click.option('-q', '--quiver', is_flag=True,
              help="Draw quiver plot.")
@click.option('-l', '--streamline', is_flag=True,
              help="Make streamline plot.")
@click.option('-s', '--scatter', is_flag=True,
              help="Plot data in scatter-plot mode.")
@click.option('--markersize', type=click.FLOAT,
              help="Set marker size for scatter plots.")
@click.option('-d', '--diverging', is_flag=True,
              help="Switch to inverted color mode.")
@click.option('-g', '--group', type=click.Choice(['0', '1']),
              help="Switch to group mode.")
@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
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
@click.option('--xlim', default=None, type=click.STRING,
              help="Set limits for the x-coordinate (lower,upper)")
@click.option('--ylim', default=None, type=click.STRING,
              help="Set limits for the y-coordinate (lower,upper).")
@click.option('--legend/--no-legend', default=True,
              help="Show legend.")
@click.option('--force-legend', 'forcelegend', is_flag=True,
              help="Force legend even when plotting a single dataset.")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
@click.option('--color', type=click.STRING,
              help="Set color when available.")
@click.option('-x', '--xlabel', type=click.STRING,
              help="Specify a x-axis label.")
@click.option('-y', '--ylabel', type=click.STRING,
              help="Specify a y-axis label.")
@click.option('-t', '--title', type=click.STRING,
              help="Specify a title.")
@click.option('--save', is_flag=True,
              help="Save figure as PNG file.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name of figure file.")
@click.option('--dpi', type=click.INT, default=200,
              help="DPI (resolution) for output")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges to show grid outline (default: None)")
@click.option('--showgrid/--no-showgrid', default=True,
              help="Show grid-lines (default: True)")
@click.option('--xkcd', is_flag=True,
              help="Turns on the xkcd style!")
@click.option('--hashtag', is_flag=True,
              help="Turns on the pgkyl hashtag!")
@click.pass_context
def plot(ctx, **kwargs):
    """Plot active datasets, optionally displaying the plot and/or saving
    it to PNG files. Plot labels can use a sub-set of LaTeX math
    commands placed between dollar ($) signs.

    """
    vlog(ctx, 'Starting plot')
    pushChain(ctx, 'plot', **kwargs)

    if kwargs['group'] is not None:
        kwargs['group'] = int(kwargs['group'])
    #end

    if kwargs['scatter']:
        kwargs['args'] = '.'
    #end

    kwargs['numAxes'] = None
    if kwargs['subplots']:
        kwargs['numAxes'] = 0
        kwargs['startAxes'] = 0
        for s in ctx.obj['sets']:
            dat = ctx.obj['dataSets'][s]
            kwargs['numAxes'] = kwargs['numAxes'] + dat.getNumComps()
        #end
        if kwargs['figure'] is None:
            kwargs['figure'] = 0
        #end
    #end
 
    fName = ""
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
        if kwargs['subplots']:
            kwargs['startAxes'] = kwargs['startAxes'] + dat.getNumComps()
        #end

        if (kwargs['save'] or kwargs['saveas']):
            if kwargs['saveas']:
                fName = kwargs['saveas']
            else:
                if fName != "":
                    fName = fName + "_"
                    #end
                if dat.fileName:
                    fName = fName + dat.fileName.split('.')[0]
                else:
                    fName = fName + 'ev_'+ctx.obj['labels'][s].replace(' ', '_')
                #end
            #end
        #end
        if (kwargs['save'] or kwargs['saveas']) and kwargs['figure'] is None:
            fName = str(fName) + '.png'
            plt.savefig(fName, dpi=kwargs['dpi'])
            fName = ""
        #end
    #end
    if (kwargs['save'] or kwargs['saveas']) and kwargs['figure'] is not None:
        fName = str(fName) + '.png'
        plt.savefig(fName, dpi=kwargs['dpi'])
    #end

    if kwargs['show']:
        plt.show()
    #end
    vlog(ctx, 'Finishing plot')
#end
