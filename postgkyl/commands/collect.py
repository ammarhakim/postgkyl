import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Collect data from the active datasets')
@click.option('-s', '--sumdata', is_flag=True,
              help="Sum data in the collected datasets (retain components)")
@click.option('-p', '--period', type=click.FLOAT,
              help="Specify a period to create epoch data instead of time data")
@click.option('-o', '--offset', default=0.0, type=click.FLOAT,
              help="Specify an offset to create epoch data instead of time data (default: 0)")
@click.option('-g', '--group', is_flag=True,
              help="Separately collect matching files into different sets")
@click.pass_context
def collect(ctx, **kwargs):
    vlog(ctx, 'Starting collect')
    pushChain(ctx, 'collect', **kwargs)
    stems = []
    activeSets = []

    group = kwargs['group']
    
    if group:
        for s in ctx.obj['sets']:
            stem = "_".join(ctx.obj['dataSets'][s].fName.split("_")[:-1])
            if not stem in stems:
                stems.append(stem)
    else:
        stems = ['']

    for st in stems:
        if group:
            vlog(ctx, 'collect: collecting files matching stem {:s}'.format(st))
        time = []
        values = []

        for s in ctx.obj['sets']:
            stem = "_".join(ctx.obj['dataSets'][s].fName.split("_")[:-1])
            if st == stem or not group:
                time.append(ctx.obj['dataSets'][s].time)
                v = ctx.obj['dataSets'][s].getValues()
                if kwargs['sumdata']:
                    numDims = ctx.obj['dataSets'][s].getNumDims()
                    axis = tuple(range(numDims))
                    values.append(np.nansum(v, axis=axis))
                else:
                    values.append(v)
        time = np.array(time)
        values = np.array(values)

        if kwargs['period'] is not None:
            time = (time - kwargs['offset']) % kwargs['period']

        sortIdx = np.argsort(time)
        time = time[sortIdx]
        values = values[sortIdx]

        if kwargs['sumdata']:
            grid = [time]
        else:
            s = ctx.obj['sets'][0]
            grid = list(ctx.obj['dataSets'][s].getGrid())
            grid.insert(0, time)

        vlog(ctx, 'collect: Creating {:d}D data with shape {}'.format(len(grid), values.shape))
        idx = len(ctx.obj['dataSets'])
        ctx.obj['setIds'].append(idx)
        ctx.obj['dataSets'].append(GData())
        ctx.obj['labels'].append('collect')
        ctx.obj['dataSets'][idx].pushGrid(grid)
        ctx.obj['dataSets'][idx].pushValues(values)
        ctx.obj['dataSets'][idx].fName = 'collect'
        vlog(ctx, 'collect: activated data set #{:d}'.format(idx))
        activeSets.append(idx)

    ctx.obj['sets'] = activeSets

    vlog(ctx, 'Finishing collect')
