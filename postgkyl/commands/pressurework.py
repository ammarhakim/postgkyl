import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
import postgkyl.diagnostics as diag

@click.command(help='Separate energy into kinetic and thermal parts, and compute pressure work')
@click.option('--gasgamma', '-g',
              default=5.0/3.0, show_default=True,
              help='Specify an adiabatic index.')
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              default='pdW', show_default=True,
              help='Tag for the resulting pressure work array')
@click.option('--label', '-l',
              default='pdW', show_default=True,
              help="Custom label for the result")
@click.pass_context
def pressurework(ctx, **kwargs):
    vlog(ctx, 'Starting pressure work computation')
    pushChain(ctx, 'pdW', **kwargs)
    data = ctx.obj['data']

    for dat in data.iterator(kwargs['use']):
        grid = dat.getGrid()
        outPdW = np.zeros(dat.getValues()[...,0:5].shape)
        grid, outPdW = diag.pressure_diagnostics.pressureWork(dat, kwargs['gasgamma'])
        dat.deactivate()
        out = Data(tag=kwargs['tag'],
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'],
                   meta=dat.meta)
        out.push(grid, outPdW)
        data.add(out)
    #end
    vlog(ctx, 'Finishing pressure work computation')
#end
