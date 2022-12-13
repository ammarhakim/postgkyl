import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import GData

@click.command(help='Integrate data over a specified axis or axes')
@click.argument('axis', nargs=1,  type=click.STRING)
@click.option('--use', '-u', default=None,
              help="Specify the tag to integrate")
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def integrate(ctx, **kwargs):
    vlog(ctx, 'Starting integrate')
    pushChain(ctx, 'integrate', **kwargs)
    data = ctx.obj['data']
    
    for dat in data.iterator(kwargs['use']):
        if kwargs['tag']:
            grid, values = diag.integrate(dat, kwargs['axis'])
            out = GData(tag=kwargs['tag'],
                        label=kwargs['label'],
                        comp_grid=ctx.obj['compgrid'],
                        meta=dat.meta)
            out.push(grid, values)
            data.add(out)
        else:
            diag.integrate(dat, kwargs['axis'], overwrite=True)
        #en
    #end
        
    vlog(ctx, 'Finishing integrate')
#end
