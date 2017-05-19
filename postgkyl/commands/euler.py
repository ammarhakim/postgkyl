import click

def pressure(gasGamma, q):
    return (gasGamma-1)*(q[...,4] - 0.5*(q[...,1]**2+q[...,2]**2+q[...,3]**2)/q[...,0])

@click.command(help='Print data info')
@click.option('-g', help="Gas adiabatic constant", type=click.FLOAT, prompt=True)
@click.option('-v', help="Variable to plot. One of 'density', 'xmom', 'ymom', 'zmom' or 'pressure' ")
@click.pass_context
def euler(ctx, g, v):
    q = ctx.obj['data'][0].q # just for now
    if v == "density":
        ctx.obj['values'][0] = q[...,0]
    elif v == "xmom":
        ctx.obj['values'][0] = q[...,1]/q[...,0]
    elif v == "ymom":
        ctx.obj['values'][0] = q[...,2]/q[...,0]
    elif v == "zmom":
        ctx.obj['values'][0] = q[...,3]/q[...,0]
    elif v == "pressure":
        ctx.obj['values'][0] = pressure(g, q)

    
