import click

from postgkyl.diagnostics import Calculus
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Integrate data over a specified axis or axes')
@click.argument('axis', nargs=1,  type=click.STRING)
@click.pass_context
def integrate(ctx, **kwargs):
    vlog(ctx, 'Starting integrate')
    pushChain(ctx, 'calculus.integrate', **kwargs)

    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]
        calc = Calculus(data)
        calc.integrate(kwargs['axis'], stack=True)
        
    vlog(ctx, 'Finishing integrate')
