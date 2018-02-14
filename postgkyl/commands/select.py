import click
import numpy as np

import postgkyl.data.select 
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Subselect data set(s)')
@click.option('--c0', default=None,
              help="Indices for 0th coord (either int, float, or slice)")
@click.option('--c1', default=None,
              help="Indices for 1st coord (either int, float, or slice)")
@click.option('--c2', default=None,
              help="Indices for 2nd coord (either int, float, or slice)")
@click.option('--c3', default=None,
              help="Indices for 3rd coord (either int, float, or slice)")
@click.option('--c4', default=None,
              help="Indices for 4th coord (either int, float, or slice)")
@click.option('--c5',  default=None,
              help="Indices for 5th coord (either int, float, or slice)")
@click.option('--comp', '-c', default=None,
              help="Indices for components (either int, slice, or coma-separated)")
@click.pass_context
def select(ctx, **kwargs):
    vlog(ctx, 'Starting select')
    pushChain(ctx, 'select.select', **kwargs)
    for s in ctx.obj['sets']:
       postgkyl.data.select(ctx.obj['dataSets'][s],
                            coord0=kwargs['c0'], coord1=kwargs['c1'],
                            coord2=kwargs['c2'], coord3=kwargs['c3'],
                            coord4=kwargs['c4'], coord5=kwargs['c5'],
                            comp=kwargs['comp'])
    vlog(ctx, 'Finishing select')

@click.command(help='Select data sets(s)')
@click.option('-i', '--idx', type=click.STRING,
              help='Data set indices')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def dataset(ctx, **kwargs):
    idx = kwargs['idx']
    if kwargs['allsets']:
        vlog(ctx, 'Selecting all datasets')
    else:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
    pushChain(ctx, 'select.dataset', **kwargs)

    if kwargs['allsets'] is False:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
        if len(idx.split(',')) > 1:
            sets = idx.split(',')
            ctx.obj['sets'] = [int(s) for s in sets]            
        elif len(idx.split(':')) == 2:
            sets = idx.split(':')
            ctx.obj['sets'] = range(int(sets[0]), int(sets[1]))
        else:
            ctx.obj['sets'] = [int(idx)]
    else:
        vlog(ctx, 'Selecting all data sets'.format(idx))
        ctx.obj['sets'] = range(ctx.obj['numSets'])

@click.command(help='Collect data from the active datasets')
@click.option('-s', '--sumdata', is_flag=True,
              help='Sum data in collected datasets')
@click.pass_context
def collect(ctx, **kwargs):
    vlog(ctx, 'Starting collect')
    pushChain(ctx, 'select.collect', **kwargs)
    gridOut = []
    valuesOut = []

    for s in ctx.obj['sets']:
        grid, values = peakStack(ctx, s)
        gridOut.append(ctx.obj['data'][s].time)
        if sumdata:
            valuesOut.append(values.sum())
        else:
            valuesOut.append(values)
    gridOut = np.array(gridOut)
    valuesOut = np.array(valuesOut)
    if kwargs['sumdata']:
        gridOut = np.expand_dims(gridOut, axis=0)
        valuesOut = np.expand_dims(valuesOut, axis=1)
    else:
        # Python 3:
        # gridOut = np.array([gridOut, *grid])
        temp = []
        temp.append(gridOut)
        for c in grid:
            temp.append(c)
        gridOut = np.array(temp)

    vlog(ctx, 
         'collect: Creating {:d}D data with shape {}'.format(len(gridOut), 
                                                             valuesOut.shape))
    vlog(ctx, 
         'collect: Active data set switched to #{:d}'.format(ctx.obj['numSets']))
    dataSet = addStack(ctx)
    ctx.obj['type'].append('hist')
    pushStack(ctx, dataSet, gridOut, valuesOut, 'collect')
    ctx.obj['sets'] = [dataSet]

    vlog(ctx, 'Finishing collect')
