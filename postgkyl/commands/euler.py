import click
import numpy as np

from postgkyl.tools.stack import pushStack, peakStack, popStack, antiSqueeze
from postgkyl.commands.output import vlog

def pressure(gasGamma, q):
    return (gasGamma-1)*(q[..., 4] -
                         0.5*(q[..., 1]**2 + q[..., 2]**2 +
                              q[..., 3]**2)/q[..., 0])

@click.command(help='Extract Euler (five-moment) primitive variables from fluid simulation')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressure"]))
@click.pass_context
def euler(ctx, gas_gamma, variable_name):
    vlog(cxt, 'Starting euler')
    v = variable_name
    for s in ctx.obj['sets']:
        coords, q = peakStack(ctx, s)

        vlog(ctx, 'euler: Extracting {:s} from data set #{:d}'.format(v, s))
        if v == "density":
            tmp = q[..., 0]
        elif v == "xvel":
            tmp = q[..., 1] / q[..., 0]
        elif v == "yvel":
            tmp = q[..., 2] / q[..., 0]
        elif v == "zvel":
            tmp = q[..., 3] / q[..., 0]
        elif v == "vel":
            tmp = np.copy(q[..., 1:4])
            tmp[..., 0] = tmp[..., 0] / q[..., 0]
            tmp[..., 1] = tmp[..., 1] / q[..., 0]
            tmp[..., 2] = tmp[..., 2] / q[..., 0]
        elif v == "pressure":
            tmp = pressure(gas_gamma, q)
        tmp = antiSqueeze(coords, tmp)

        pushStack(ctx, s, coords, tmp, v)
    vlog(ctx, 'Finishing euler')

    
