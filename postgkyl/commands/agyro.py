import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

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

def getForb(pij, B):
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

    ppar = bx*bx*pxx + by*by*pyy + bz*bz*pzz + 2.0*(bx*by*pxy + bx*bz*pxz + by*bz*pyz)
    pper = (pxx + pyy + pzz - ppar)/2.0

    pixx = pxx - (ppar*bx*bx+pper*(1-bx*bx)) # xx
    pixy = pxy - (ppar*bx*by+pper*(0-bx*by)) # xy
    pixz = pxz - (ppar*bx*bz+pper*(0-bx*bz)) # xz
    piyy = pyy - (ppar*by*by+pper*(1-by*by)) # yy
    piyz = pyz - (ppar*by*bz+pper*(0-by*bz)) # yz
    pizz = pzz - (ppar*bz*bz+pper*(1-bz*bz)) # zz
    
    return np.sqrt(pixx**2+2*pixy**2+2*pixz**2+piyy**2+2*piyz**2+pizz**2)/np.sqrt(2*pper**2+4*ppar*pper)

@click.command(help=r'Compute a measure of agyrotropy. Default measure is taken from Swisdak 2015. Optionally computes agyrotropy as Frobenius norm of agyrotropic pressure tensor. Pressure-tensor must be the first dataset and magnetic field the second dataset.')
@click.option('--forb', is_flag=True, default=False,
              help='Compute agyrotropy using Forbenius norm.')
@click.pass_context
def agyro(ctx, **kwargs):
    vlog(ctx, 'Starting agyro')
    pushChain(ctx, 'agyro', **kwargs)

    grid = ctx.obj['dataSets'][ctx.obj['sets'][0]].peakGrid()
    lo, up = ctx.obj['dataSets'][ctx.obj['sets'][0]].getBounds()

    pij = ctx.obj['dataSets'][ctx.obj['sets'][0]].peakValues()
    B = ctx.obj['dataSets'][ctx.obj['sets'][1]].peakValues()

    if kwargs['forb']:
        tmp = getForb(pij, B)
    else:
        tmp = getSwisdak(pij, B)

    tmp = tmp[..., np.newaxis]

    ctx.obj['dataSets'].append(GData())
    idx = len(ctx.obj['dataSets'])-1
    ctx.obj['dataSets'][idx].pushGrid(grid, lo, up)
    ctx.obj['dataSets'][idx].pushValues(tmp)
    ctx.obj['sets'] = [idx]

    vlog(ctx, 'Finishing agyro')

    
