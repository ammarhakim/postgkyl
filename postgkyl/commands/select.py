import click
import numpy

from postgkyl.tools.fields import fixCoordSlice
from postgkyl.tools.stack import pushStack, peakStack, popStack

@click.command(help='Fix a coordinate')
@click.option('--c1', type=click.FLOAT, help='Fix 1st coordinate')
@click.option('--c2', type=click.FLOAT, help='Fix 2nd coordinate')
@click.option('--c3', type=click.FLOAT, help='Fix 3rd coordinate')
@click.option('--c4', type=click.FLOAT, help='Fix 4th coordinate')
@click.option('--c5', type=click.FLOAT, help='Fix 5th coordinate')
@click.option('--c6', type=click.FLOAT, help='Fix 6th coordinate')
@click.option('--value', 'mode', flag_value='value',
              default=True, help='Fix coordinates based on a value')
@click.option('--index', 'mode', flag_value='idx',
              help='Fix coordinates based on an index')
@click.pass_context
def fix(ctx, c1, c2, c3, c4, c5, c6, mode):
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)
        coordsOut, valuesOut = fixCoordSlice(coords, values, mode,
                                             c1, c2, c3, c4, c5, c6)
        pushStack(ctx, s, coordsOut, valuesOut)

@click.command(help='Select component(s)')
@click.argument('component', type=click.STRING)
@click.pass_context
def comp(ctx, component):
    component = component.split(',')
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)

        if len(component) == 1:
            comps = int(component[0])
            values = values[..., comps, numpy.newaxis]
        else:
            comps = slice(int(component[0]), int(component[1])+1)
            values = values[..., comps]

        pushStack(ctx, s, coords, values)

@click.command(help='Pop the data stack')
@click.pass_context
def pop(ctx):
    for s in range(ctx.obj['numSets']):
        popStack(ctx, s)
