import click

from postgkyl.data.interp import GInterpNodalSerendipity
from postgkyl.data.interp import GInterpModalSerendipity
from postgkyl.data.interp import GInterpModalMaxOrder

@click.command()
@click.option('--basis', '-b',
              type=click.Choice(['ns', 'ms', 'mo']))
@click.option('--polyorder', '-p', type=click.INT)
@click.pass_context
def project(ctx, basis, polyorder):
    ctx.obj['coords'] = []
    ctx.obj['values'] = []
    for i in range(ctx.obj['numFiles']):
        if basis == 'ns':
            dg = GInterpNodalSerendipity(ctx.obj['data'][i], polyorder)
        elif basis == 'ms':
            dg = GInterpModalSerendipity(ctx.obj['data'][i], polyorder)
        elif basis == 'mo':
            dg = GInterpModalMaxOrder(ctx.obj['data'][i], polyorder)
        coords, values = dg.project(0)
        ctx.obj['coords'].append(coords)
        ctx.obj['values'].append(values)
        
              
