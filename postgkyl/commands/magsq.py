import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
import postgkyl.diagnostics as diag

@click.command(help='Magnitude squared of an input array |A|^2 = sum_i A_i^2')
@click.option('--tag', '-t', default=None,
              help="Specify the tag to integrate")
@click.pass_context
def magsq(ctx, **kwargs):
    """Calculate the magnitude squared of an input array
    """
    vlog(ctx, 'Starting magnitude squared computation')
    pushChain(ctx, 'magsq', **kwargs)
    data = ctx.obj['data']
    for dat in data.iterator(kwargs['tag']):
        diag.magsq(dat, stack=True)
    #end
        
    vlog(ctx, 'Finishing magnitude squared computation')
#end
