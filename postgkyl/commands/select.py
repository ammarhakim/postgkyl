import click
import numpy

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
    for i, values in enumerate(ctx.obj['values']):
        coords, values = fixCoordSlice(ctx.obj['coords'][i],
                                       ctx.obj['values'][i],
                                       mode,
                                       c1, c2, c3, c4, c5, c6)
        ctx.obj['coords'][i] = coords
        ctx.obj['values'][i] = values

@click.command(help='Select component(s)')
@click.argument('component', type=click.STRING)
@click.pass_context
def comp(ctx, component):
    components = component.split(',')
    numSets = ctx.obj['numSets']
    for s in range(numSets):
        if len(components) == 1:
            comps = slice(int(components[0]), int(components[0])+1)
        else:
            comps = slice(int(components[0]), int(components[1])+1)
        ctx.obj['mapComps'][s] = comps
        ctx.obj['numComps'][s] = len(components)
