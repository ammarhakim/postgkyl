import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Integrate data over a specified axis or axes')
@click.argument('axis', nargs=1,  type=click.STRING)
@click.option('--tag', '-t', default=None,
              help="Specify the tag to integrate")
@click.pass_context
def integrate(ctx, **kwargs):
    vlog(ctx, 'Starting integrate')
    pushChain(ctx, 'integrate', **kwargs)
    data = ctx.obj['data']
    for dat in data.iterator(kwargs['tag']):
        diag.integrate(dat, kwargs['axis'], stack=True)
    #end
        
    vlog(ctx, 'Finishing integrate')
#end
