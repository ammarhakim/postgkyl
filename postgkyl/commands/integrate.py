import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Integrate data over a specified axis or axes')
@click.argument('axis', nargs=1,  type=click.STRING)
@click.pass_context
def integrate(ctx, **kwargs):
    vlog(ctx, 'Starting integrate')
    pushChain(ctx, 'integrate', **kwargs)

    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]
        diag.integrate(data, kwargs['axis'], stack=True)
    #end
        
    vlog(ctx, 'Finishing integrate')
#end
