import click
import numpy as np

from postgkyl.data import Data
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('-s', '--sumdata', is_flag=True,
              help="Sum data in the collected datasets (retain components)")
@click.option('-p', '--period', type=click.FLOAT,
              help="Specify a period to create epoch data instead of time data")
@click.option('-o', '--offset', default=0.0, type=click.FLOAT,
              help="Specify an offset to create epoch data instead of time data (default: 0)")
@click.option('-c', '--chunk', type=click.INT,
              help="Collect into chunks with specified length rather than into a single dataset")
@click.option('-g', '--group', is_flag=True,
              help="Separately collect matching files into different sets")
@click.pass_context
def collect(ctx, **kwargs):
    """Collect data from the active datasets and create a new combined
    dataset. The time-stamp in each of the active datasets is
    collected and used as the new X-axis. Data can be collected in
    chunks, in which case several datasets are created, each with the
    chunk-sized pieces collected into each new dataset.

    """
    vlog(ctx, 'Starting collect')
    pushChain(ctx, 'collect', **kwargs)
    stems = []
    activeSets = []

    group = kwargs['group']
    
    if group:
        for s in ctx.obj['sets']:
            stem = "_".join(ctx.obj['dataSets'][s].fileName.split("_")[:-1])
            if not stem in stems:
                stems.append(stem)
            #end
        #end
    else:
        stems = ['collect']
    #end

    for st in stems:
        if group:
            vlog(ctx, 'collect: collecting files matching stem {:s}'.format(st))
        #end
        time = [[]]
        values = [[]]
        chunkIdx = 0
        cnt = 0

        for s in ctx.obj['sets']:
            stem = "_".join(ctx.obj['dataSets'][s].fileName.split("_")[:-1])
            if st == stem or not group:
                # we are looping through all sets, and if group, only want to work on ones matching stem
                cnt = cnt + 1
                if kwargs['chunk'] is not None and cnt > kwargs['chunk']:
                    chunkIdx = chunkIdx + 1
                    cnt = 1
                    time.append([])
                    values.append([])
                #end
                time[chunkIdx].append(ctx.obj['dataSets'][s].time)
                v = ctx.obj['dataSets'][s].getValues()
                if kwargs['sumdata']:
                    numDims = ctx.obj['dataSets'][s].getNumDims()
                    axis = tuple(range(numDims))
                    values[chunkIdx].append(np.nansum(v, axis=axis))
                else:
                    values[chunkIdx].append(v)
                #end

                # need to assign grid in this block so that each stem group has its own grid
                grid = list(ctx.obj['dataSets'][s].getGrid())
            #end
        #end

        for i in range(len(time)):
            time[i] = np.array(time[i])
            values[i] = np.array(values[i])

            if kwargs['period'] is not None:
                time[i] = (time[i] - kwargs['offset']) % kwargs['period']
            #end

            sortIdx = np.argsort(time[i])
            time[i] = time[i][sortIdx]
            values[i] = values[i][sortIdx]

            if kwargs['sumdata']:
                grid = [time[i]]
            else:
                grid.insert(0, time[i])
            #end

            vlog(ctx, 'collect: Creating {:d}D data with shape {}'.format(len(grid), values[i].shape))
            idx = len(ctx.obj['dataSets'])
            ctx.obj['setIds'].append(idx)
            ctx.obj['dataSets'].append(Data())
            ctx.obj['labels'].append(st)
            ctx.obj['dataSets'][idx].pushGrid(grid)
            ctx.obj['dataSets'][idx].pushValues(values[i])
            ctx.obj['dataSets'][idx].time = 0.5*(time[i][0]+time[i][-1])
            ctx.obj['dataSets'][idx].fileName = st
            vlog(ctx, 'collect: activated data set #{:d}'.format(idx))
            activeSets.append(idx)
        #end
    #end
    ctx.obj['sets'] = activeSets

    vlog(ctx, 'Finishing collect')
#end
