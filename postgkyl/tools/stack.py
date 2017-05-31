import numpy

from postgkyl.data.load import GData, GHistoryData
from postgkyl.data.interp import GInterpZeroOrder

def pushStack(ctx, dataSet, coords, values, label=''):
    ctx.obj['coords'][dataSet].append(coords)
    ctx.obj['values'][dataSet].append(values)
    ctx.obj['labels'][dataSet].append(label)  
def peakStack(ctx, dataSet):
    coords = ctx.obj['coords'][dataSet][-1]
    values = ctx.obj['values'][dataSet][-1]
    return coords, values
def popStack(ctx, dataSet):
    coords = ctx.obj['coords'][dataSet].pop()
    values = ctx.obj['values'][dataSet].pop()
    ctx.obj['labels'][dataSet].pop()
    return coords, values

def peakLabel(ctx, dataSet, idx=None):
    if idx is not None:
        return ctx.obj['labels'][dataSet][idx]
    else:
        cnt = -1
        while True:
            label = ctx.obj['labels'][dataSet][cnt]
            if label != '':
                return label
            cnt -= 1

def getFullLabel(ctx, dataSet, joinStr='_'):
    labels = ctx.obj['labels'][dataSet][:]
    for label in labels:
        if label == '':
            labels.remove('')
    return  joinStr.join(labels)

def antiSqueeze(coords, values):
    numDims = len(coords)
    for d in range(numDims):
        if len(coords[d]) == 1:
            values = numpy.expand_dims(values, axis=d)

    if len(values.shape) == numDims:
        values = values[..., numpy.newaxis]

    return values

def loadFrame(ctx, dataSet, fileName):
    ctx.obj['data'].append(GData(fileName))

    dg = GInterpZeroOrder(ctx.obj['data'][dataSet])
    coords, values = dg.project(0)
    values = antiSqueeze(coords, values)

    numDims = ctx.obj['data'][dataSet].numDims
    numComps = int(ctx.obj['data'][dataSet].q.shape[-1])

    if numComps > 1:
        for c in numpy.arange(numComps-1)+1:
            coords, tmp = dg.project(c)
            values = numpy.append(values, antiSqueeze(coords, tmp),
                                  axis=numDims)

    name = fileName.split('/')[-1]  # get rid of the full path
    name = ''.join(name.split('.')[: -1])  # get rid of the extension
    # This weird Python construct is here in case someone would
    # like to use '.' in name... I really dislike it but I don't
    # know about any better -pc

    ctx.obj['coords'].append([])
    ctx.obj['values'].append([])
    ctx.obj['labels'].append([])
    pushStack(ctx, dataSet, coords, values, name)

def loadHist(ctx, dataSet, fileName):
    ctx.obj['data'].append('')
    hist = GHistoryData(fileName)

    name = fileName.split('/')[-1]  # get rid of the full path
    name = name.strip('0')
    name = name.strip('_')

    ctx.obj['coords'].append([])
    ctx.obj['values'].append([])
    ctx.obj['labels'].append([])
    pushStack(ctx, dataSet,
              hist.time[numpy.newaxis, ...], hist.values[..., numpy.newaxis],
              name)
