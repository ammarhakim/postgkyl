import click
import numpy as np
import tm

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze
from postgkyl.commands.output import vlog

def getParPerp(pij, B):
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

    tmp[...,0] = bx*bx*pxx + by*by*pyy + bz*bz*pzz + 2.0*(bx*by*pxy + bx*bz*pxz + by*bz*pyz)
    tmp[...,1] = (pxx + pyy + pzz - tmp[...,0])/2.0

    return tmp

@click.command(help=r'Extract parallel and perpendicular pressures from pressure-tensor and magnetic field. Pressure-tensor must be the first dataset and magnetic field the second dataset. A two component field (parallel, perpendicular) is returned.')
@click.pass_context
def cglpressure(ctx):
    vlog(ctx, 'Starting CGL pressure')

    coords, pij = peakStack(ctx, ctx.obj['sets'][0])
    coords, B = peakStack(ctx, ctx.obj['sets'][1])
    
    tmp = getParPerp(pij, B)
    tmp = antiSqueeze(coords, tmp)

    pushStack(ctx, 0, coords, tmp, 'CGL')
    vlog(ctx, 'Finishing CGL pressure')

    
