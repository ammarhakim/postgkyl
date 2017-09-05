import click
import numpy as np

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze
from postgkyl.commands.output import vlog

def getRho(q):
    return q[...,0]
def getU(q):
    return q[...,1]/q[...,0]
def getV(q):
    return q[...,2]/q[...,0]
def getW(q):
    return q[...,3]/q[...,0]

def getPxx(q):
    r = getRho(q)
    u = getU(q)
    return q[...,4] - r*u*u

def getPxy(q):
    r = getRho(q)
    u = getU(q)
    v = getV(q)
    return q[...,5] - r*u*v

def getPxz(q):
    r = getRho(q)
    u = getU(q)
    w = getW(q)
    return q[...,6] - r*u*w

def getPyy(q):
    r = getRho(q)
    v = getV(q)
    return q[...,7] - r*v*v

def getPyz(q):
    r = getRho(q)
    v = getV(q)
    w = getW(q)
    return q[...,8] - r*v*w

def getPzz(q):
    r = getRho(q)
    w = getW(q)
    return q[...,9] - r*w*w

def getVel(q):
    tmp = np.copy(q[..., 1:4])
    tmp[..., 0] = getU(q)
    tmp[..., 1] = getV(q)
    tmp[..., 2] = getW(q)
    return tmp

def getPressureTensor(q):
    tmp = np.copy(q[..., 4:10])
    tmp[..., 0] = getPxx(q)
    tmp[..., 1] = getPxy(q)
    tmp[..., 2] = getPxz(q)
    tmp[..., 3] = getPyy(q)
    tmp[..., 4] = getPyz(q)
    tmp[..., 5] = getPzz(q)
    return tmp

def getPressure(q):
    return (getPxx(q)+getPyy(q)+getPzz(q))/3.0

@click.command(help='Extract ten-moment primitive variables from fluid simulation')
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressureTensor",
                                 "pxx", "pxy", "pxz", "pyy", "pyz", "pzz",
                                 "pressure"
              ]))
@click.pass_context
def tenmoment(ctx, variable_name):
    vlog(ctx, 'Starting tenmoment')
    v = variable_name
    for s in ctx.obj['sets']:
        coords, q = peakStack(ctx, s)

        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            tmp = getRho(q)
        elif v == "xvel":
            tmp = getU(q)
        elif v == "yvel":
            tmp = getV(q)
        elif v == "zvel":
            tmp = getW(q)
        elif v == "vel":
            tmp = getVel(q)
        elif v == "pxx":
            tmp = getPxx(q)
        elif v == "pxy":
            tmp = getPxy(q)
        elif v == "pxz":
            tmp = getPxz(q)
        elif v == "pyy":
            tmp = getPyy(q)
        elif v == "pyz":
            tmp = getPyz(q)
        elif v == "pzz":
            tmp = getPzz(q)
        elif v == "pressure":
            tmp = getPressure(q)
        elif v == "pressureTensor":
            tmp = getPressure(q)
        else:
            vlog(ctx, 'No such variable %s' % v)
            
        tmp = antiSqueeze(coords, tmp)

        pushStack(ctx, s, coords, tmp, v)
    vlog(ctx, 'Finishing tenmoment')

    
