import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('--density', '-d',
              default='density', show_default=True,
              help="Tag for density")
@click.option('--momentum', '-m',
              default='momentum', show_default=True,
              help="Tag for momentum")
@click.option('--outtag', '-o',
              default='velocity', show_default=True,
              help='Tag for the result')
@click.option('--label', '-l',
              default='velocity', show_default=True,
              help="Custom label for the result")
@click.pass_context
def velocity(ctx, **kwargs):
    vlog(ctx, 'Starting velocity')
    pushChain(ctx, 'velocity', **kwargs)
    
    data = ctx.obj['data'] # shortcut

    # Check for correct tags
    if not kwargs['density'] in data.tagIterator():
        ctx.fail(click.style("Failed to load the specified/default tag '{:s}'".format(kwargs['density']),
                             fg='red'))
    #end
    if not kwargs['momentum'] in data.tagIterator():
        ctx.fail(click.style("Failed to load the specified/default tag '{:s}'".format(kwargs['momentum']),
                             fg='red'))
    #end
    
    for m0, m1 in zip(data.iterator(kwargs['density']),
                      data.iterator(kwargs['momentum'])):
        grid = m0.getGrid()        
        valsM0 = m0.getValues()
        valsM1 = m1.getValues()
            
        out = Data(tag=kwargs['outtag'],
                   stack=ctx.obj['stack'],
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'])
        out.push(valsM1/valsM0, grid)
        data.add(out)
    #end

    vlog(ctx, 'Finishing velocity')
#end
