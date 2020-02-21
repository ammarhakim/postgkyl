import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Mask data with specified Gkeyll mask file.')
@click.argument('filenm', nargs=1,  type=click.STRING)
@click.pass_context
def mask(ctx, **kwargs):
    vlog(ctx, 'Starting mask')
    pushChain(ctx, 'mask', **kwargs)

    maskFld = GData(kwargs['filenm']).getValues()

    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]
        grid = list(data.getGrid())
        values = data.getValues()

        data.pushGrid(grid)
        maskFldRep = np.repeat(maskFld, data.getNumComps(), axis=-1)
        data.pushValues(np.ma.masked_where(maskFldRep < 0.0, values))
    #end
        
    vlog(ctx, 'Finishing mask')
#end
