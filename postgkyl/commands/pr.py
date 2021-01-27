import click
import numpy as np
np.set_printoptions(precision=16)

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print the data')
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.pass_context
def pr(ctx, **kwargs):
    vlog(ctx, 'Starting pr')
    pushChain(ctx, 'pr', **kwargs)
    data = ctx.obj['data']
    for dat in data.iterator(kwargs['tag']):
        click.echo(dat.getValues().squeeze())
    #end

    vlog(ctx, 'Finishing pr')
#end
