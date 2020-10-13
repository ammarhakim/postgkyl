import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain

#---------------------------------------------------------------------
#-- Math -------------------------------------------------------------
@click.command(help='Multiply data by a factor')
@click.argument('factor', nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, **kwargs):
    vlog(ctx, 'Multiplying by {:f}'.format(kwargs['factor']))
    pushChain(ctx, 'temp.mult', **kwargs)
    for s in ctx.obj['sets']:
        values = ctx.obj['dataSets'][s].getValues()
        values = values * kwargs['factor']
        ctx.obj['dataSets'][s].push(values)
    #end
#end

@click.command(help='Calculate power of data')
@click.argument('power', nargs=1, type=click.FLOAT)
@click.pass_context
def pow(ctx, **kwargs):
    vlog(ctx, 'Calculating the power of {:f}'.format(kwargs['power']))
    pushChain(ctx, 'temp.pow', **kwargs)
    for s in ctx.obj['sets']:
        values = ctx.obj['dataSets'][s].getValues()
        values = values ** kwargs['power']
        ctx.obj['dataSets'][s].push(values)
    #end
#end

@click.command(help='Calculate natural log of data')
@click.pass_context
def log(ctx):
    vlog(ctx, 'Calculating the natural log')
    pushChain(ctx, 'temp.log')
    for s in ctx.obj['sets']:
        values = ctx.obj['dataSets'][s].getValues()
        values = np.log(values)
        ctx.obj['dataSets'][s].push(values)
    #end
#end    

@click.command(help='Calculate absolute values of data')
@click.pass_context
def abs(ctx):
    vlog(ctx, 'Calculating the absolute value')
    pushChain(ctx, 'transform.log')
    for s in ctx.obj['sets']:
        values = ctx.obj['dataSets'][s].getValues()
        values = np.abs(values)
        ctx.obj['dataSets'][s].push(values)
    #end
#end

@click.command(help='Normalize data')
@click.option('--shift/--no-shift', default=False,
              help='Shift minimal value to zero (default: False).')
@click.option('--usefirst', is_flag=True,  default=False,
              help='Normalize to first value in field')
@click.pass_context
def norm(ctx, **kwargs):
    vlog(ctx, 'Normalizing data')
    pushChain(ctx, 'temp.norm', **kwargs)
    for s in ctx.obj['sets']:
        values = ctx.obj['dataSets'][s].getValues()
        numComps = ctx.obj['dataSets'][s].getNumComps()
        valuesOut = values.copy()
        for comp in range(numComps):
            if kwargs['shift']:
                valuesOut[..., comp] -= valuesOut[..., comp].min()
            if kwargs["usefirst"]:
                valuesOut[..., comp] /= valuesOut[..., comp].item(0)
            else:
                valuesOut[..., comp] /= np.abs(valuesOut[..., comp]).max()
            #end
        #end
        ctx.obj['dataSets'][s].push(valuesOut)
    #end
#end

# @click.command(help='Calculate gradient')
# @click.pass_context
# def grad(ctx):
#     vlog(ctx, 'Calculating gradient')
#     pushChain(ctx, 'transform.grad')
#     for s in ctx.obj['sets']:
#         values = peakStack(ctx, s)
#         numDims = len(coords)
#         numComps = values.shape[-1]
#         if numComps > 1:
#             click.echo('Warning - grad: scalar value expecded, only the first component is taken into the account')

#         valuesOut = np.gradient(values[..., 0], edge_order=2)
#         valuesOut = np.moveaxis(valuesOut, 0, -1)
#         for d in range(numDims):
#             valuesOut[..., d] /= (coords[d][1]-coords[d][0])

#         pushStack(ctx, s, coords, valuesOut, 'grad')

# @click.command(help='Calculate divergence')
# @click.pass_context
# def div(ctx):
#     vlog(ctx, 'Calculating divergence')
#     pushChain(ctx, 'transform.div')
#     for s in ctx.obj['sets']:
#         coords, values = peakStack(ctx, s)

#         numDims = len(coords)
#         numComps = values.shape[-1]
#         if numDims != numComps:
#             raise ValueError(
#                 "div: number of dimensions is not corresponding to the number of components")

#         dx = coords[0][1]-coords[0][0]
#         valuesOut = antiSqueeze(coords, np.gradient(values[..., 0], dx,
#                                                     axis=0, edge_order=2))
#         for d in np.arange(numDims-1)+1:
#             d = int(d)
#             dx = coords[d][1]-coords[d][0]
#             temp = antiSqueeze(coords, np.gradient(values[..., d], dx,
#                                                    axis=d, edge_order=2))
#             valuesOut += temp

#         pushStack(ctx, s, coords, valuesOut, 'div')

# @click.command(help='Calculate curl')
# @click.pass_context
# def curl(ctx):
#     vlog(ctx, 'Calculating curl')
#     pushChain(ctx, 'transform.curl')
#     for s in ctx.obj['sets']:
#         coords, values = peakStack(ctx, s)

#         numDims = len(coords)
#         numComps = values.shape[-1]
#         if numComps != 3:
#             raise ValueError(
#                 "curl: 3 componets (3D vector) are required for curl")

#         dx = coords[0][1]-coords[0][0]
#         dy = coords[1][1]-coords[1][0]
#         dudy = np.gradient(values[..., 0], dy, axis=1, edge_order=2)
#         dvdx = np.gradient(values[..., 1], dx, axis=0, edge_order=2)
#         if numDims == 3:
#             dwdx = np.gradient(values[..., 2], dx, axis=0, edge_order=2)
#             dwdy = np.gradient(values[..., 2], dy, axis=1, edge_order=2)
#             dz = coords[2][1]-coords[2][0]
#             dudz = np.gradient(values[..., 0], dz, axis=2, edge_order=2)
#             dvdz = np.gradient(values[..., 1], dz, axis=2, edge_order=2)

#         if numDims == 2:
#             valuesOut = np.zeros(values.shape[:-1])
#             valuesOut = dvdx - dudy
#             valuesOut = valuesOut[..., np.newaxis]
#         else: 
#             valuesOut = np.zeros(values.shape)
#             valuesOut[..., 0] = dwdy - dvdz
#             valuesOut[..., 1] = dudz - dwdx
#             valuesOut[..., 2] = dvdx - dudy
        
#         pushStack(ctx, s, coords, valuesOut, 'curl')


# #---------------------------------------------------------------------
# #-- Miscellaneous ----------------------------------------------------
# @click.command(help='Mask data')
# @click.argument('maskfile', nargs=1, type=click.STRING)
# @click.pass_context
# def mask(ctx, **inputs):
#     vlog(ctx, 'Masking data with {:s}'.format(inputs['maskField']))
#     pushChain(ctx, 'transform.mask', **inputs)

#     maskField = GData(inputs['maskfile']).q[..., 0, np.newaxis]
#     for s in ctx.obj['sets']:
#         coords, values = peakStack(ctx, s)

#         numComps = values.shape[-1]
#         numDims = len(values.shape) - 1

#         tmp = np.copy(maskField)
#         if numComps > 1:
#             for comp in np.arange(numComps-1)+1:
#                 tmp = np.append(tmp, maskField, axis=numDims)

#         valuesOut = np.ma.masked_where(tmp < 0.0, values)

#         pushStack(ctx, s, coords, valuesOut)
