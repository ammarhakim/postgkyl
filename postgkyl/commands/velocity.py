import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('--density', '-d', default=None,
              help="Specify the tag for density.")
@click.option('--momentum', '-m', default=None,
              help="Specify the tag for momentum.")
@click.option('--tag', '-t', default='velocity',
              help="Specify the custom tag for the output.")
@click.option('--label', '-l', default='velocity',
              help="Specify the custom label for the output.")
@click.pass_context
def velocity(ctx, **kwargs):
    vlog(ctx, 'Starting velocity')
    pushChain(ctx, 'velocity', **kwargs)

    for m0, m1 in zip(ctx.obj['data'].iterator(kwargs['density']),
                      ctx.obj['data'].iterator(kwargs['momentum'])):
        grid = m0.getGrid()        
        valsM0 = m0.getValues()
        valsM1 = m1.getValues()

        out = Data(tag=kwargs['tag'],
                   stack=ctx.obj['stack'],
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'])
        out.push(valsM1/valsM0, grid=grid)
        ctx.obj['data'].add(out)
    #end

    vlog(ctx, 'Finishing velocity')
#end
