import click
import numpy as np

from postgkyl.tools.fields import fixGridSlice
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Fix a coordinate')
@click.option('--c0', type=click.FLOAT, help='Fix 1st coordinate')
@click.option('--c1', type=click.FLOAT, help='Fix 2nd coordinate')
@click.option('--c2', type=click.FLOAT, help='Fix 3rd coordinate')
@click.option('--c3', type=click.FLOAT, help='Fix 4th coordinate')
@click.option('--c4', type=click.FLOAT, help='Fix 5th coordinate')
@click.option('--c5', type=click.FLOAT, help='Fix 6th coordinate')
@click.option('--value', 'mode', flag_value='value',
              default=True, help='Fix coordinates based on a value')
@click.option('--index', 'mode', flag_value='idx',
              help='Fix coordinates based on an index')
@click.pass_context
def fix(ctx, **kwargs):
    vlog(ctx, 'Starting fix')
    pushChain(ctx, 'select.fix', **kwargs)

    for s in ctx.obj['sets']:
        grid, values = ctx.obj['dataSets'][s].peakStack()
        gridOut, valuesOut = fixGridSlice(grid, values, kwargs['mode'],
                                          kwargs['c0'], kwargs['c1'],
                                          kwargs['c2'], kwargs['c3'],
                                          kwargs['c4'], kwargs['c5'])
        label = 'fix'
        if kwargs['c0'] is not None:
            label = '{:s}_c0_{:f}'.format(label, kwargs['c0'])
        if kwargs['c1'] is not None:
            label = '{:s}_c1_{:f}'.format(label, kwargs['c1'])
        if kwargs['c2'] is not None:
            label = '{:s}_c2_{:f}'.format(label, kwargs['c2'])
        if kwargs['c3'] is not None:
            label = '{:s}_c3_{:f}'.format(label, kwargs['c3'])
        if kwargs['c4'] is not None:
            label = '{:s}_c4_{:f}'.format(label, kwargs['c4'])
        if kwargs['c5'] is not None:
            label = '{:s}_c5_{:f}'.format(label, kwargs['c5'])

        #pushStack(ctx, s, gridOut, valuesOut, label)
        ctx.obj['dataSets'][s].pushStack(gridOut, valuesOut)

@click.command(help='Select component(s)')
@click.argument('component', type=click.STRING)
@click.pass_context
def comp(ctx, **kwargs):
    vlog(ctx, 'Selecting component(s): {:s}'.format(kwargs['component']))
    pushChain(ctx, 'select.comp', **kwargs)

    for s in ctx.obj['sets']:
        grid, values = ctx.obj['dataSets'][s].peakStack()

        component = kwargs['component']
        if len(component.split(',')) > 1:
            components = component.split(',')
            label = 'c_{:s}'.format('_'.join(components)) 
            idx = [int(c) for c in components]
            values = values[..., tuple(idx)]
        elif len(component.split(':')) == 2:
            components = component.split(':')
            label = 'c_{:s}'.format('-'.join(components))
            comps = slice(int(components[0]), int(components[1]))
            values = values[..., comps]
        else:
            comp = int(component)
            label = 'c_{:d}'.format(comp)
            values = values[..., comp, np.newaxis]

        #pushStack(ctx, s, grid, values, label)
        ctx.obj['dataSets'][s].pushStack(grid, values)

@click.command(help='Select data sets(s)')
@click.option('-i', '--idx', type=click.STRING,
              help='Data set indices')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def dataset(ctx, **kwargs):
    if kwargs['allsets']:
        vlog(ctx, 'Selecting all datasets')
    else:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
    pushChain(ctx, 'select.dataset', **kwargs)

    idx = kwargs['idx']
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

@click.command(help='Pop the data stack')
@click.pass_context
def pop(ctx):
    vlog(ctx, 'Poping the stack')
    pushChain(ctx, 'select.pop')
    for s in ctx.obj['sets']:
        popStack(ctx, s)

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
