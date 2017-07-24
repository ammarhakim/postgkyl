import click
import numpy as np
import matplotlib.pyplot as plt

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze

#---------------------------------------------------------------------
#-- Growth -----------------------------------------------------------
@click.command(help='Fit exponential to data')
@click.option('-g', '--guess', default=(1.0, 0.1),
              help='Specify initial guess')
@click.option('-p', '--plot', is_flag=True,
              help='Plot the data and fit')
@click.option('--minn', default=100, type=click.INT,
              help='Set minimal number of points to fit')
@click.option('--maxn', type=click.INT,
              help='Set maximal number of points to fit')
@click.pass_context
def growth(ctx, guess, plot, minn, maxn):
    from postgkyl.diagnostics.growth import fitGrowth, exp2

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        numDims = len(coords)
        numComps = values.shape[-1]

        bestParams, bestR2, bestN = fitGrowth(coords[0], values[..., 0],
                                              minN=minn, maxN=maxn,
                                              p0=guess)

        if plot is True:
            plt.style.use(ctx.obj['mplstyle'])
            fig, ax = plt.subplots()
            ax.plot(coords[0], values[..., 0], '.')
            ax.set_autoscale_on(False)
            ax.plot(coords[0], exp2(coords[0], *bestParams))
            ax.grid(True)
            plt.show()
