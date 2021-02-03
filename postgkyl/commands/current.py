import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
import postgkyl.diagnostics.accumulate_current

@click.command(help='Accumulate current, sum over species of charge multiplied by flow')
@click.option('--qbym', '-q',
              default=False, show_default=True,
              help="Flag for multiplying by charge/mass ratio instead of just charge")
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--outtag', '-o',
              default='current', show_default=True,
              help='Tag for the resulting current array')
@click.option('--label', '-l',
              default='J', show_default=True,
              help="Custom label for the result")
@click.pass_context
def current(ctx, **kwargs):
    vlog(ctx, 'Starting current accumulation')
    pushChain(ctx, 'current', **kwargs)
    data = ctx.obj['data']

    for dat in data.iterator(kwargs['tag']):
        grid = dat.getGrid()
        outcurrent = np.zeros(dat.getValues().shape)
        outcurrent += postgkyl.diagnostics.accumulate_current(dat, kwargs['qbym'])
        dat.deactivate()
        out = Data(tag=kwargs['outtag'],
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'],
                   meta=dat.meta)
        out.push(grid, outcurrent)
        data.add(out)
    #end
    vlog(ctx, 'Finishing current accumulation')
#end
