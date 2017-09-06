import click
import numpy as np
import tm

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze, addStack
from postgkyl.commands.output import vlog

def getSwisdak(pij, B):
    tmp = np.copy(pij[..., 0:2])

    pxx = pij[...,0]
    pxy = pij[...,1]
    pxz = pij[...,2]
    pyy = pij[...,3]
    pyz = pij[...,4]
    pzz = pij[...,5]

    b = np.sqrt(B[...,0]*B[...,0] + B[...,1]*B[...,1] + B[...,2]*B[...,2])
    bx = B[...,0]/b
    by = B[...,1]/b
    bz = B[...,2]/b

    pPar = bx*bx*pxx + by*by*pyy + bz*bz*pzz + 2.0*(bx*by*pxy + bx*bz*pxz + by*bz*pyz)
    I1 = pxx + pyy + pzz
    I2 = pxx*pyy + pxx*pzz + pyy*pzz - (pxy*pxy + pxz*pxz + pyz*pyz)
    return np.sqrt(1 - 4*I2/((I1 - pPar)*(I1 + 3*pPar)))

@click.command(help=r'Compute a measure of agyrotropy using measure from Swisdak 2015. Optionally computes agyrotropy as deviation of local magnetic field direction from pressure tensor principle axes. Pressure-tensor must be the first dataset and magnetic field the second dataset.')
@click.pass_context
def agyro(ctx):
    vlog(ctx, 'Starting agyro')

    coords, pij = peakStack(ctx, ctx.obj['sets'][0])
    coords, B = peakStack(ctx, ctx.obj['sets'][1])

    tmp = getSwisdak(pij, B)
    tmp = antiSqueeze(coords, tmp)

    idx = addStack(ctx)
    ctx.obj['type'].append('hist')
    pushStack(ctx, idx, coords, tmp, 'agyro')
    ctx.obj['sets'] = [idx]

    vlog(ctx, 'Finishing agyro')

    
