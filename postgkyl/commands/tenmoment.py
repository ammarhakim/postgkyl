import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressureTensor",
                                 "pxx", "pxy", "pxz", "pyy", "pyz", "pzz",
                                 "pressure"
              ]))
@click.pass_context
def tenmoment(ctx, **kwargs):
    """Extract ten-moment primitive variables from ten-moment conserved
    variables.

    """
    vlog(ctx, 'Starting tenmoment')
    pushChain(ctx, 'tenmoment', **kwargs)

    v = kwargs['variable_name']
    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]

        vlog(ctx, 'tenmoment: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            diag.getDensity(data, stack=True)
        elif v == "xvel":
            diag.getVx(data, stack=True)
        elif v == "yvel":
            diag.getVy(data, stack=True)
        elif v == "zvel":
            diag.getVz(data, stack=True)
        elif v == "vel":
            diag.getVi(data, stack=True)
        elif v == "pxx":
            diag.getPxx(data, stack=True)
        elif v == "pxy":
            diag.getPxy(data, stack=True)
        elif v == "pxz":
            diag.getPxz(data, stack=True)
        elif v == "pyy":
            diag.getPyy(data, stack=True)
        elif v == "pyz":
            diag.getPyz(data, stack=True)
        elif v == "pzz":
            diag.getPzz(data, stack=True)
        elif v == "pressure":
            diag.getP(data, numMoms=10, stack=True)
        elif v == "pressureTensor":
            diag.getPij(data, stack=True)

    vlog(ctx, 'Finishing tenmoment')

    
