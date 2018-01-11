import click
import numpy as np

from postgkyl.tools.fields import fixCoordSlice
from postgkyl.tools.stack import addStack, pushStack, peakStack, popStack
from postgkyl.commands.output import vlog, pushChain

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
def fix(ctx, **inputs):
    vlog(ctx, 'Starting fix')
    pushChain(ctx, 'select.fix', **inputs)

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        coordsOut, valuesOut = fixCoordSlice(coords, values, inputs['mode'],
                                             inputs['c0'], inputs['c1'],
                                             inputs['c2'], inputs['c3'],
                                             inputs['c4'], inputs['c5'])
        label = 'fix'
        if inputs['c0'] is not None:
            label = '{:s}_c0_{:f}'.format(label, inputs['c0'])
        if inputs['c1'] is not None:
            label = '{:s}_c1_{:f}'.format(label, inputs['c1'])
        if inputs['c2'] is not None:
            label = '{:s}_c2_{:f}'.format(label, inputs['c2'])
        if inputs['c3'] is not None:
            label = '{:s}_c3_{:f}'.format(label, inputs['c3'])
        if inputs['c4'] is not None:
            label = '{:s}_c4_{:f}'.format(label, inputs['c4'])
        if inputs['c5'] is not None:
            label = '{:s}_c5_{:f}'.format(label, inputs['c5'])

        pushStack(ctx, s, coordsOut, valuesOut, label)

@click.command(help='Select component(s)')
@click.argument('component', type=click.STRING)
@click.pass_context
def comp(ctx, **inputs):
    vlog(ctx, 'Selecting component(s): {:s}'.format(inputs['component']))
    pushChain(ctx, 'select.comp', **inputs)

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        component = inputs['component']
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

        pushStack(ctx, s, coords, values, label)       

@click.command(help='Select data sets(s)')
@click.option('-i', '--idx', type=click.STRING,
              help='Data set indices')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def dataset(ctx, **inputs):
    if inputs['allsets']:
        vlog(ctx, 'Selecting all datasets')
    else:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
    pushChain(ctx, 'select.dataset', **inputs)

    idx = inputs['idx']
    if inputs['allsets'] is False:
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
def collect(ctx, **inputs):
    vlog(ctx, 'Starting collect')
    pushChain(ctx, 'select.collect', **inputs)
    coordsOut = []
    valuesOut = []

    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        coordsOut.append(ctx.obj['data'][s].time)
        if sumdata:
            valuesOut.append(values.sum())
        else:
            valuesOut.append(values)
    coordsOut = np.array(coordsOut)
    valuesOut = np.array(valuesOut)
    if inputs['sumdata']:
        coordsOut = np.expand_dims(coordsOut, axis=0)
        valuesOut = np.expand_dims(valuesOut, axis=1)
    else:
        # Python 3:
        # coordsOut = np.array([coordsOut, *coords])
        temp = []
        temp.append(coordsOut)
        for c in coords:
            temp.append(c)
        coordsOut = np.array(temp)

    vlog(ctx, 
         'collect: Creating {:d}D data with shape {}'.format(len(coordsOut), 
                                                             valuesOut.shape))
    vlog(ctx, 
         'collect: Active data set switched to #{:d}'.format(ctx.obj['numSets']))
    dataSet = addStack(ctx)
    ctx.obj['type'].append('hist')
    pushStack(ctx, dataSet, coordsOut, valuesOut, 'collect')
    ctx.obj['sets'] = [dataSet]

    vlog(ctx, 'Finishing collect')
