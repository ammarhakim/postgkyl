import click
import numpy

from postgkyl.data.load import GData
from postgkyl.data.interp import GInterpNodalSerendipity
from postgkyl.data.interp import GInterpModalSerendipity
from postgkyl.data.interp import GInterpModalMaxOrder

from postgkyl.tools.fields import fixCoordSlice

@click.command(help='Project DG data on a uniform mesh')
@click.option('--basis', '-b', prompt=True,
              type=click.Choice(['ns', 'ms', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', prompt=True, type=click.INT,
              help='Specify polynomial order')
@click.pass_context
def project(ctx, basis, polyorder):
    for i in range(ctx.obj['numFiles']):
        if basis == 'ns':
            dg = GInterpNodalSerendipity(ctx.obj['data'][i], polyorder)
        elif basis == 'ms':
            dg = GInterpModalSerendipity(ctx.obj['data'][i], polyorder)
        elif basis == 'mo':
            dg = GInterpModalMaxOrder(ctx.obj['data'][i], polyorder)
        coords, values = dg.project(0)
        ctx.obj['coords'][i] = coords
        ctx.obj['values'][i] = values

@click.command(help='Multiply data by a factor')
@click.argument('factor', nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, factor):
    for i, values in enumerate(ctx.obj['values']):
        ctx.obj['values'][i] = values * factor

@click.command(help='Normalize data')
@click.option('--shift/--no-shift', default=False,
              help='Shift minimal value to zero (default: False).')
@click.pass_context
def norm(ctx, shift):
    for i, values in enumerate(ctx.obj['values']):
        if shift:
            values -= values.min() 
        ctx.obj['values'][i] = values / numpy.abs(values).max()

@click.command(help='Mask data')
@click.argument('maskfile', nargs=1, type=click.STRING)
@click.pass_context
def mask(ctx, maskfile):
    maskField = GData(maskfile).q[..., 0]
    for i, values in enumerate(ctx.obj['values']):
        ctx.obj['values'][i] = numpy.ma.masked_where(maskField < 0.0, values)

@click.command(help='Fix a coordinate')
@click.option('-c1', type=click.FLOAT, help='Fix 1st coordinate')
@click.option('-c2', type=click.FLOAT, help='Fix 2nd coordinate')
@click.option('-c3', type=click.FLOAT, help='Fix 3rd coordinate')
@click.option('-c4', type=click.FLOAT, help='Fix 4th coordinate')
@click.option('-c5', type=click.FLOAT, help='Fix 5th coordinate')
@click.option('-c6', type=click.FLOAT, help='Fix 6th coordinate')
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
