import click

def pressure(gasGamma, q):
    return (gasGamma-1)*(q[...,4] - 0.5*(q[...,1]**2+q[...,2]**2+q[...,3]**2)/q[...,0])

@click.command(help='Extract Euler (five-moment) primitive variables from fluid simulation')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant", type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to plot.", prompt=True,
              type=click.Choice(["density", "xvel", "yvel", "zvel", "pressure"]))
@click.pass_context
def euler(ctx, gas_gamma, variable_name):
    v = variable_name
    for i in range(len(ctx.obj['values'])):
        q = ctx.obj['values'][i]
        if v == "density":
            ctx.obj['values'][i] = q[...,0]
        elif v == "xvel":
            ctx.obj['values'][i] = q[...,1]/q[...,0]
        elif v == "yvel":
            ctx.obj['values'][i] = q[...,2]/q[...,0]
        elif v == "zvel":
            ctx.obj['values'][i] = q[...,3]/q[...,0]
        elif v == "pressure":
            ctx.obj['values'][i] = pressure(gas_gamma, q)

    
