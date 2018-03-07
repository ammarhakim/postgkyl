import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Extract Euler (five-moment) primitive variables from fluid simulation')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressure"]))
@click.pass_context
def euler(ctx, **kwargs):
    vlog(ctx, 'Starting euler')
    pushChain(ctx, 'euler', **kwargs)

    v = kwargs['variable_name']
    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]

        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
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
        elif v == "pressure":
            diag.getP(data, gasGamma=kwargs['gas_gamma'], numMoms=5, stack=True)

    vlog(ctx, 'Finishing euler')

    
