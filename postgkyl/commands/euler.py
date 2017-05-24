import click
import numpy

from postgkyl.tools.stack import pushStack, pullStack, popStack

def pressure(gasGamma, q):
    return (gasGamma-1)*(q[...,4] -
                         0.5*(q[...,1]**2+q[...,2]**2+q[...,3]**2)/q[...,0])

@click.command(help='Extract Euler (five-moment) primitive variables from fluid simulation')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "pressure"]))
@click.pass_context
def euler(ctx, gas_gamma, variable_name):
    v = variable_name
    for s in range(ctx.obj['numSets']):
        coords, q = pullStack(ctx, s)

        if v == "density":
            tmp = q[...,0]
        elif v == "xvel":
            tmp = q[...,1]/q[...,0]
        elif v == "yvel":
            tmp = q[...,2]/q[...,0]
        elif v == "zvel":
            tmp = q[...,3]/q[...,0]
        elif v == "pressure":
            tmp = pressure(gas_gamma, q)
        tmp = tmp[..., numpy.newaxis]

        pushStack(ctx, s, coords, tmp)

    
