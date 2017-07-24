import click
import numpy as np

from postgkyl.data.load import GData
from postgkyl.data.interp import GInterpNodalSerendipity
from postgkyl.data.interp import GInterpModalSerendipity
from postgkyl.data.interp import GInterpModalMaxOrder
from postgkyl.data.interp import GInterpGeneral
from postgkyl.data.interp import GInterpGeneralRead

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze

#---------------------------------------------------------------------
#-- DG projection ----------------------------------------------------
@click.command(help='Project DG data on a uniform mesh')
@click.option('--basis', '-b', prompt=True,
              type=click.Choice(['ns', 'ms', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', prompt=True, type=click.INT,
              help='Specify polynomial order')
@click.option('--general', '-g', type=click.INT,
              help='Interpolation onto a general mesh of specified amount')
@click.option('--read-general', '-r', type=click.BOOL,
              help='Interpolation onto a general mesh of specified amount')
@click.pass_context
def project(ctx, basis, polyorder, general, read_general):
    for s in ctx.obj['sets']:
        data = ctx.obj['data'][s]
        numDims = data.numDims

        if general is not None:
            interp = general
            dg = GInterpGeneral(data, polyorder, basis, interp)
            if basis == 'ns':
                numNodes = GInterpNodalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
            elif basis == 'ms':
                numNodes = GInterpModalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
            elif basis == 'mo':
                numNodes = GInterpModalMaxOrder.numNodes[numDims-1,
                                                     polyorder-1] 
        elif read_general is not None:
            dg = GInterpGeneralRead(data, polyorder, basis)
            if basis == 'ns':
                numNodes = GInterpNodalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
            elif basis == 'ms':
                numNodes = GInterpModalSerendipity.numNodes[numDims-1,
                                                        polyorder-1]
            elif basis == 'mo':
                numNodes = GInterpModalMaxOrder.numNodes[numDims-1,
                                                     polyorder-1] 
        else:
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
            for comp in np.arange(numComps-1)+1:
                coords, tmp = dg.project(comp)
                values = np.append(values, antiSqueeze(coords, tmp),
                                      axis=numDims)
 
        label = 'proj_{:s}_{:d}'.format(basis, polyorder)
        pushStack(ctx, s, coords, values, label)


#---------------------------------------------------------------------
#-- Math -------------------------------------------------------------
@click.command(help='Multiply data by a factor')
@click.argument('factor', nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, factor):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        valuesOut = values * factor
        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Calculate power of data')
@click.argument('power', nargs=1, type=click.FLOAT)
@click.pass_context
def pow(ctx, power):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        valuesOut = values**power
        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Calculate natural log of data')
@click.pass_context
def log(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        valuesOut = np.log(values)
        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Calculate absolute values of data')
@click.pass_context
def abs(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        valuesOut = np.abs(values)
        pushStack(ctx, s, coords, valuesOut)

@click.command(help='Normalize data')
@click.option('--shift/--no-shift', default=False,
              help='Shift minimal value to zero (default: False).')
@click.pass_context
def norm(ctx, shift):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        
        numComps = values.shape[-1]
        valuesOut = values.copy()
        for comp in range(numComps):
            if shift:
                valuesOut[..., comp] -= valuesOut[..., comp].min() 
            valuesOut[..., comp] /= np.abs(valuesOut[..., comp]).max()  

        label = 'norm'
        pushStack(ctx, s, coords, valuesOut, label)

@click.command(help='Transpose data')
@click.pass_context
def transpose(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)
        numDims = len(coords)
        numComps = values.shape[-1]

        if numDims == 2:
            coordsOut = np.array([coords[1], coords[0]])
            valuesOut = np.zeros((values.shape[1],
                                  values.shape[0],
                                  values.shape[2]))
            for comp in range(numComps):
                valuesOut[..., comp] = values[..., comp].transpose()

            pushStack(ctx, s, coordsOut, valuesOut, 'trans')
        else:
            click.echo('Warning - transpose: transpose only available for 2D data; ignoring')

#---------------------------------------------------------------------
#-- Calculus ---------------------------------------------------------
@click.command(help='Integrate over axes')
@click.argument('axis', nargs=1, type=click.STRING)
@click.pass_context
def integrate(ctx, axis):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        axes = axis.split(',')
        label = 'int_{:s}'.format('_'.join(axes)) 
        axes = [int(a) for a in axes]

        valuesOut = np.sum(values, axis=tuple(axes))
        for a in axes:
            valuesOut *= (coords[a][1] - coords[a][0])

        numDims = len(coords)
        idxCoords = []
        for d in range(numDims):
            if not d in axes:
                idxCoords.append(d)
        coordsOut = coords[idxCoords]

        pushStack(ctx, s, coordsOut, valuesOut, label)

@click.command(help='Calculate gradient')
@click.pass_context
def grad(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        numDims = len(coords)
        numComps = values.shape[-1]
        if numComps > 1:
            click.echo('Warning - grad: scalar value expecded, only the first component is taken into the account')

        valuesOut = np.gradient(values[..., 0], edge_order=2)
        valuesOut = np.moveaxis(valuesOut, 0, -1)
        for d in range(numDims):
            valuesOut[..., d] /= (coords[d][1]-coords[d][0])

        pushStack(ctx, s, coords, valuesOut, 'grad')

@click.command(help='Calculate divergence')
@click.pass_context
def div(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        numDims = len(coords)
        numComps = values.shape[-1]
        if numDims != numComps:
            raise ValueError(
                "div: number of dimensions is not corresponding to the number of components")

        dx = coords[0][1]-coords[0][0]
        valuesOut = antiSqueeze(coords, np.gradient(values[..., 0], dx,
                                                    axis=0, edge_order=2))
        for d in np.arange(numDims-1)+1:
            d = int(d)
            dx = coords[d][1]-coords[d][0]
            temp = antiSqueeze(coords, np.gradient(values[..., d], dx,
                                                   axis=d, edge_order=2))
            valuesOut += temp

        pushStack(ctx, s, coords, valuesOut, 'div')

@click.command(help='Calculate curl')
@click.pass_context
def curl(ctx):
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        numDims = len(coords)
        numComps = values.shape[-1]
        if numComps != 3:
            raise ValueError(
                "curl: 3 componets (3D vector) are required for curl")

        dx = coords[0][1]-coords[0][0]
        dy = coords[1][1]-coords[1][0]
        dudy = np.gradient(values[..., 0], dy, axis=1, edge_order=2)
        dvdx = np.gradient(values[..., 1], dx, axis=0, edge_order=2)
        if numDims == 3:
            dwdx = np.gradient(values[..., 2], dx, axis=0, edge_order=2)
            dwdy = np.gradient(values[..., 2], dy, axis=1, edge_order=2)
            dz = coords[2][1]-coords[2][0]
            dudz = np.gradient(values[..., 0], dz, axis=2, edge_order=2)
            dvdz = np.gradient(values[..., 1], dz, axis=2, edge_order=2)

        if numDims == 2:
            valuesOut = np.zeros(values.shape[:-1])
            valuesOut = dvdx - dudy
            valuesOut = valuesOut[..., np.newaxis]
        else: 
            valuesOut = np.zeros(values.shape)
            valuesOut[..., 0] = dwdy - dvdz
            valuesOut[..., 1] = dudz - dwdx
            valuesOut[..., 2] = dvdx - dudy
        
        pushStack(ctx, s, coords, valuesOut, 'curl')


#---------------------------------------------------------------------
#-- Miscellaneous ----------------------------------------------------
@click.command(help='Mask data')
@click.argument('maskfile', nargs=1, type=click.STRING)
@click.pass_context
def mask(ctx, maskfile):
    maskField = GData(maskfile).q[..., 0, np.newaxis]
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s)

        numComps = values.shape[-1]
        numDims = len(values.shape) - 1

        tmp = np.copy(maskField)
        if numComps > 1:
            for comp in np.arange(numComps-1)+1:
                tmp = np.append(tmp, maskField, axis=numDims)

        valuesOut = np.ma.masked_where(tmp < 0.0, values)

        pushStack(ctx, s, coords, valuesOut)

#---------------------------------------------------------------------
#-- Transformations --------------------------------------------------
@click.command(help="Calculate Fast Fourier Transform")
@click.option('-p', '--psd', is_flag=True,
              help='Return real power density rather that complex FFT')
@click.pass_context
def fft(ctx, psd):
    from scipy import fftpack
    for s in ctx.obj['sets']:
        coords, values = peakStack(ctx, s) 
        
        numComps = values.shape[-1]
        numDims = values.ndim - 1
        if numDims > 1:
            click.echo('fft: 1D data required for FFT')
            ctx.exit()
        N = len(coords[0])

        dx = coords[0][1] - coords[0][0]
        coordsOut = fftpack.fftfreq(N, dx)
        valuesOut = np.zeros(values.shape, 'complex')
        for comp in np.arange(numComps):
            valuesOut[..., comp] = fftpack.fft(values[..., comp])

        if psd:
            coordsOut = coordsOut[:N//2]
            valuesOut = np.abs(valuesOut[:N//2, :])**2

        coordsOut = np.expand_dims(coordsOut, axis=0)
        pushStack(ctx, s, coordsOut, valuesOut, 'fft')

#---------------------------------------------------------------------
#-- Filters ----------------------------------------------------------
