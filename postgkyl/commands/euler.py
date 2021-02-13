import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
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
    for dat in data.iterator(kwargs['use']):
        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            diag.getDensity(dat, overwrite=True)
        elif v == "xvel":
            diag.getVx(dat, overwrite=True)
        elif v == "yvel":
            diag.getVy(dat, overwrite=True)
        elif v == "zvel":
            diag.getVz(dat, overwrite=True)
        elif v == "vel":
            diag.getVi(dat, overwrite=True)
        elif v == "pressure":
            diag.getP(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
        elif v == "ke":
            diag.getKE(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
        elif v == "mach":
            diag.getMach(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
        #end
    vlog(ctx, 'Finishing euler')
#end
    
