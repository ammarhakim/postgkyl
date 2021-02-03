import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Integrate data over a specified axis or axes')
@click.argument('axis', nargs=1,  type=click.STRING)
@click.option('--tag', '-t', default=None,
              help="Specify the tag to integrate")
@click.option('--outtag', '-o',
              help='Optional tag for the resulting array')
@click.pass_context
def integrate(ctx, **kwargs):
    vlog(ctx, 'Starting integrate')
    pushChain(ctx, 'integrate', **kwargs)
    data = ctx.obj['data']
    for dat in data.iterator(kwargs['tag']):
        if kwargs['outtag']:
            grid, values = diag.integrate(dat, kwargs['axis'])
            out = Data(tag=kwargs['outtag'],
                       compgrid=ctx.obj['compgrid'],
                       meta=dat.meta)
            out.push(grid, values)
            data.add(out)
        else:
            diag.integrate(dat, kwargs['axis'], overwrite=True)
        #en
    #end
        
    vlog(ctx, 'Finishing integrate')
#end
