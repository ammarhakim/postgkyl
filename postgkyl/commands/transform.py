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
    for s in range(ctx.obj['numSets']):
        data = ctx.obj['data'][s]
        numDims = data.numDims
        if basis == 'ns':
            dg = GInterpNodalSerendipity(data, polyorder)
            numNodes = GInterpNodalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
        elif basis == 'ms':
            dg = GInterpModalSerendipity(data, polyorder)
            numNodes = GInterpModalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
        elif basis == 'mo':
            dg = GInterpModalMaxOrder(data, polyorder)
            numNodes = GInterpModalMaxOrder.numNodes[numDims-1,
                                                     polyorder-1]
        coords, values = dg.project(0)
        values = numpy.expand_dims(values, axis=numDims)
        numComps = int(data.q.shape[-1]/numNodes)
        ctx.obj['numComps'][s] = numComps
        if numComps > 1:
            for comp in numpy.arange(numComps-1)+1:
                coords, tmp = dg.project(comp)
                tmp = numpy.expand_dims(tmp, axis=numDims)
                values = numpy.append(values, tmp, axis=numDims)
        ctx.obj['coords'][s] = coords
        ctx.obj['values'][s] = values
        mapValues = [slice(0, values.shape[d]) for d in range(numDims)]
        ctx.obj['mapValues'][s] = mapValues
        ctx.obj['mapComps'][s] = slice(0, numComps)

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
