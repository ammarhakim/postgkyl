import click
import numpy

from postgkyl.data.load import GData
from postgkyl.data.interp import GInterpNodalSerendipity
from postgkyl.data.interp import GInterpModalSerendipity
from postgkyl.data.interp import GInterpModalMaxOrder

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze

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
        values = antiSqueeze(coords, values)

        numComps = int(data.q.shape[-1]/numNodes)
        if numComps > 1:
            for comp in numpy.arange(numComps-1)+1:
                coords, tmp = dg.project(comp)
                values = numpy.append(values, antiSqueeze(coords, tmp),
                                      axis=numDims)
 
        label = 'proj_{:s}_{:d}'.format(basis, polyorder)
        pushStack(ctx, s, coords, values, label)

@click.command(help='Multiply data by a factor')
@click.argument('factor', nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, factor):
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)
        valuesOut = values * factor
        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Normalize data')
@click.option('--shift/--no-shift', default=False,
              help='Shift minimal value to zero (default: False).')
@click.pass_context
def norm(ctx, shift):
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)
        
        numComps = values.shape[-1]
        valuesOut = values.copy()
        for comp in range(numComps):
            if shift:
                valuesOut[..., comp] -= valuesOut[..., comp].min() 
            valuesOut[..., comp] /= numpy.abs(valuesOut[..., comp]).max()  

        label = 'norm'
        pushStack(ctx, s, coords, valuesOut, label)

@click.command(help='Mask data')
@click.argument('maskfile', nargs=1, type=click.STRING)
@click.pass_context
def mask(ctx, maskfile):
    maskField = GData(maskfile).q[..., 0, numpy.newaxis]
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)

        numComps = values.shape[-1]
        numDims = len(values.shape) - 1

        tmp = numpy.copy(maskField)
        if numComps > 1:
            for comp in numpy.arange(numComps-1)+1:
                tmp = numpy.append(tmp, maskField, axis=numDims)

        valuesOut = numpy.ma.masked_where(tmp < 0.0, values)

        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Integrate over axies')
@click.argument('axies', nargs=1, type=click.STRING)
@click.pass_context
def integrate(ctx, axies):
    for s in range(ctx.obj['numSets']):
        coords, values = peakStack(ctx, s)

        axies = axies.split(',')
        label = 'int_{:s}'.format('_'.join(axies)) 
        axies = [int(axis) for axis in axies]

        valuesOut = numpy.sum(values, axis=tuple(axies))
        for axis in axies:
            valuesOut *= (coords[axis][1] - coords[axis][0])

        numDims = len(coords)
        idxCoords = []
        for d in range(numDims):
            if not d in axies:
                idxCoords.append(d)
        if len(axies) == numDims-1:
             coordsOut = coords[numpy.newaxis, idxCoords]
        else:
            coordsOut = coords[idxCoords]

        pushStack(ctx, s, coordsOut, valuesOut, label)
