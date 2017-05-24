import numpy

from postgkyl.data.load import GData, GHistoryData
from postgkyl.data.interp import GInterpZeroOrder

def pushStack(ctx, dataSet, coords, values):
    ctx.obj['coords'][dataSet].append(coords)
    ctx.obj['values'][dataSet].append(values)
def pullStack(ctx, dataSet):
    coords = ctx.obj['coords'][dataSet][-1]
    values = ctx.obj['values'][dataSet][-1]
    return coords, values
def popStack(ctx, dataSet):
    coords = ctx.obj['coords'][dataSet].pop()
    values = ctx.obj['values'][dataSet].pop()
    return coords, values

def loadFrame(ctx, dataSet, fileName):
    ctx.obj['data'].append(GData(fileName))

    dg = GInterpZeroOrder(ctx.obj['data'][dataSet])
    coords, values = dg.project(0)
    values = values[..., numpy.newaxis]

    numDims = ctx.obj['data'][dataSet].numDims
    numComps = int(ctx.obj['data'][dataSet].q.shape[-1])

    if numComps > 1:
        for c in numpy.arange(numComps-1)+1:
            coords, tmp = dg.project(c)
            values = numpy.append(values, tmp[..., numpy.newaxis],
                                  axis=numDims)

    pushStack(ctx, dataSet, coords, values)

def loadHist(ctx, dataSet, fileName):
    ctx.obj['data'].append('')
    hist = GHistoryData(fileName)

    pushStack(ctx, dataSet,
              hist.time[ numpy.newaxis, ...], hist.values[..., numpy.newaxis])
