import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to extract", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressure", "ke", "mach"]))
@click.pass_context
def euler(ctx, **kwargs):
    """Compute Euler (five-moment) primitive and some derived variables
    from fluid conserved variables.

    """
    vlog(ctx, 'Starting euler')
    pushChain(ctx, 'euler', **kwargs)
    data = ctx.obj['data']
    
    v = kwargs['variable_name']
    for dat in data.iterator(kwargs['tag']):
        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
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
        elif v == "pressure":
            diag.getP(dat, gasGamma=kwargs['gas_gamma'], numMom=5, stack=True)
        elif v == "ke":
            diag.getKE(dat, gasGamma=kwargs['gas_gamma'], numMom=5, stack=True)
        elif v == "mach":
            diag.getMach(dat, gasGamma=kwargs['gas_gamma'], numMom=5, stack=True)
        #end
    vlog(ctx, 'Finishing euler')
#end
    
