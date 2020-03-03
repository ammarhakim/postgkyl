import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Mask data with specified Gkeyll mask file.')
@click.option('--filename', '-f', type=click.STRING,
              help="Specify the file with a mask")
@click.option('--lower', '-l', type=click.FLOAT,
              help="Specify the lower theshold to be masked out.")
@click.option('--upper', '-u', type=click.FLOAT,
              help="Specify the upper theshold to be masked out.")
@click.pass_context
def mask(ctx, **kwargs):
    vlog(ctx, 'Starting mask')
    pushChain(ctx, 'mask', **kwargs)

    if kwargs['filename']:
        maskFld = GData(kwargs['filename']).getValues()
    #end

    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]
        grid = list(data.getGrid())
        values = data.getValues()

        data.pushGrid(grid)

        if kwargs['filename']:
            maskFldRep = np.repeat(maskFld, data.getNumComps(), axis=-1)
            data.pushValues(np.ma.masked_where(maskFldRep < 0.0, values))
        elif kwargs['lower'] is not None and kwargs['upper'] is not None:
            data.pushValues(np.ma.masked_outside(values, kwargs['lower'], kwargs['upper']))
        elif kwargs['lower'] is not None:
            data.pushValues(np.ma.masked_less(values, kwargs['lower']))
        elif kwargs['upper'] is not None:
            data.pushValues(np.ma.masked_greater(values, kwargs['upper']))
        else:
            data.pushValues(values)
            click.echo(click.style("WARNING in 'mask': No masking information specified.", fg='yellow'))
        #end
    #end
        
    vlog(ctx, 'Finishing mask')
#end
