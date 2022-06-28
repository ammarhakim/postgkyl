import os
import click
import numpy as np
import matplotlib.pyplot as plt

from postgkyl.commands.util import vlog, pushChain
from postgkyl.diagnostics.growth import fitGrowth, exp2
from postgkyl.data import Data

#---------------------------------------------------------------------
#-- Growth -----------------------------------------------------------
@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-g', '--guess',
              help='Specify comma-separated initial guess')
@click.option('--minn', type=click.INT,
              help='Set minimal number of points to fit')
@click.option('-d', '--dataset', is_flag=True,
              help='Create a new dataset with fitted exponential')
@click.option('-i', '--instantaneous', is_flag=True,
              help='Plot instantaneous growth rate vs time')
@click.pass_context
def growth(ctx, **kwargs):
  """Attempts to compute growth rate (i.e. fit e^(2x)) from DynVector
  data, typically an integrated quantity like electric or magnetic
  field energy.
  """
  vlog(ctx, 'Starting growth')
  pushChain( ctx, 'growth', **kwargs) 
  data = ctx.obj['data']
    
  for dat in data.iterator(kwargs['use']):
    time = dat.getGrid()
    values = dat.getValues()
    numDims = dat.getNumDims()
    if numDims > 1:
      click.fail(
        click.style("'growth' is available only for 1D data (used on {:d}D data)".format(numDims),
                    fg='red'))
    #end
    p0 = kwargs['guess']
    if kwargs['guess']:
      guess = kwargs['guess'].split(',')
      p0 = (float(guess[0]), float(guess[1]))
    #end
    bestParams, bestR2, bestN = fitGrowth(time[0], values[..., 0],
                                          minN=kwargs['minn'],
                                          p0=p0)

    if kwargs['dataset']:
      out = Data(tag='growth',
                 label='Fit',
                 comp_grid=ctx.obj['compgrid'],
                 meta=dat.meta)
      t = 0.5*(time[0][:-1] + time[0][1:])
      out_val = exp2(t, *bestParams)
      out.push(time, out_val[..., np.newaxis])
      data.add(out)
    #end

    if kwargs['instantaneous'] is True:
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
    #end
  #end
  vlog(ctx, 'Finishing growth')
#end
