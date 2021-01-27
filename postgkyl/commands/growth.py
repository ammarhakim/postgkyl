import os
import click
import numpy as np
import matplotlib.pyplot as plt

from postgkyl.commands.util import vlog, pushChain
from postgkyl.diagnostics.growth import fitGrowth, exp2

#---------------------------------------------------------------------
#-- Growth -----------------------------------------------------------
@click.command()
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-g', '--guess', default=(1.0, 0.1),
              help='Specify initial guess')
@click.option('-p', '--plot', is_flag=True,
              help='Plot the data and fit')
@click.option('--minn', default=100, type=click.INT,
              help='Set minimal number of points to fit')
@click.option('--maxn', type=click.INT,
              help='Set maximal number of points to fit')
@click.option('-i', '--instantaneous', is_flag=True,
              help='Plot instantaneous growth rate vs time')
@click.pass_context
def growth(ctx, **inputs):
    """Attempts to compute growth rate (i.e. fit e^(2x)) from DynVector
    data, typically an integrated quantity like electric or magnetic
    field energy.
    """
    vlog(ctx, 'Starting growth')
    pushChain( ctx, 'growth', **inputs) 
    data = ctx.obj['data']
    
    for dat in data.iterator(kwargs['key']):
        time = dat.getGrid()
        values = dat.getValues()
        numDims = dat.getNumDims()
        if numDims > 1:
            click.echo(click.style("ERROR: 'growth' is available only for 1D data (used on {:d}D data)".format(numDims), fg='red'))
            ctx.exit()
        #end
        
        vlog(ctx, 'growth: Starting fit for data set #{:d}'.format(s))
        bestParams, bestR2, bestN = fitGrowth(time[0], values[..., 0],
                                              minN=inputs['minn'],
                                              maxN=inputs['maxn'],
                                              p0=inputs['guess'])

        if inputs['plot'] is True:
            vlog(ctx, 'growth: Plotting data and fit')
            plt.style.use(os.path.dirname(os.path.realpath(__file__)) \
                      + "/../output/postgkyl.mplstyle")
            fig, ax = plt.subplots()
            ax.plot(time[0], values[..., 0], '.')
            ax.set_autoscale_on(False)
            ax.plot(time[0], exp2(time[0], *bestParams))
            ax.grid(True)
            plt.show()
        #end

        if inputs['instantaneous'] is True:
            vlog(ctx, 'growth: Plotting instantaneous growth rate')
            gammas = []
            for i in range(1,len(time[0])-1):
                gamma = (values[i+1,0] - values[i-1,0])/(2*values[i,0]*(time[0][i+1] - time[0][i-1]))
                gammas.append(gamma)
            
            plt.style.use(os.path.dirname(os.path.realpath(__file__)) \
                      + "/../output/postgkyl.mplstyle")
            fig, ax = plt.subplots()
            ax.plot(time[0][1:-1], gammas)
            #ax.set_autoscale_on(False)
            ax.grid(True)
            plt.show()
        #end
    vlog(ctx, 'Finishing growth')
#end
