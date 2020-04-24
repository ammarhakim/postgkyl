import click
import numpy as np
np.set_printoptions(precision=16)

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print the data')
@click.pass_context
def pr(ctx, **kwargs):
    vlog(ctx, 'Starting pr')
    pushChain(ctx, 'print', **kwargs)

    for s in ctx.obj['sets']:
        print(ctx.obj['dataSets'][s].getValues().squeeze())
    #end

    vlog(ctx, 'Finishing pr')
#end
