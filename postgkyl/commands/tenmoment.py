import click
import numpy as np

from postgkyl.commands import tm
from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze
from postgkyl.commands.output import vlog, pushChain

@click.command(help='Extract ten-moment primitive variables from fluid simulation')
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressureTensor",
                                 "pxx", "pxy", "pxz", "pyy", "pyz", "pzz",
                                 "pressure"
              ]))
@click.pass_context
def tenmoment(ctx, **inputs):
    vlog(ctx, 'Starting tenmoment')
    pushChain(ctx, **inputs)

    v = inputs['variable_name']
    for s in ctx.obj['sets']:
        coords, q = peakStack(ctx, s)

        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            tmp = tm.getRho(q)
        elif v == "xvel":
            tmp = tm.getU(q)
        elif v == "yvel":
            tmp = tm.getV(q)
        elif v == "zvel":
            tmp = tm.getW(q)
        elif v == "vel":
            tmp = tm.getVel(q)
        elif v == "pxx":
            tmp = tm.getPxx(q)
        elif v == "pxy":
            tmp = tm.getPxy(q)
        elif v == "pxz":
            tmp = tm.getPxz(q)
        elif v == "pyy":
            tmp = tm.getPyy(q)
        elif v == "pyz":
            tmp = tm.getPyz(q)
        elif v == "pzz":
            tmp = tm.getPzz(q)
        elif v == "pressure":
            tmp = tm.getPressure(q)
        elif v == "pressureTensor":
            tmp = tm.getPressureTensor(q)
        else:
            vlog(ctx, 'No such variable %s' % v)
            
        tmp = antiSqueeze(coords, tmp)

        pushStack(ctx, s, coords, tmp, v)

    vlog(ctx, 'Finishing tenmoment')

    
