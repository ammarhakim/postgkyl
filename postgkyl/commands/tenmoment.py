import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
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
    data = ctx.obj['data']
    
    v = kwargs['variable_name']
    for dat in data.iterator(kwargs['use']):
        vlog(ctx, 'tenmoment: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            diag.getDensity(dat, stack=True)
        elif v == "xvel":
            diag.getVx(dat, stack=True)
        elif v == "yvel":
            diag.getVy(dat, stack=True)
        elif v == "zvel":
            diag.getVz(dat, stack=True)
        elif v == "vel":
            diag.getVi(dat, stack=True)
        elif v == "pxx":
            diag.getPxx(dat, stack=True)
        elif v == "pxy":
            diag.getPxy(dat, stack=True)
        elif v == "pxz":
            diag.getPxz(dat, stack=True)
        elif v == "pyy":
            diag.getPyy(dat, stack=True)
        elif v == "pyz":
            diag.getPyz(dat, stack=True)
        elif v == "pzz":
            diag.getPzz(dat, stack=True)
        elif v == "pressure":
            diag.getP(dat, numMoms=10, stack=True)
        elif v == "pressureTensor":
            diag.getPij(dat, stack=True)
        #end
    vlog(ctx, 'Finishing tenmoment')
#end
    
