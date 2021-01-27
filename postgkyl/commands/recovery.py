import click
import numpy as np

from postgkyl.data import GInterpModal
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Interpolate DG data on a uniform mesh')
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Number of poins to evaluate on')
@click.option('-r', '--periodic', is_flag=True,
              help='Flag for periodic boundary conditions')
@click.option('-c', '--c1', is_flag=True,
              help='Enforce continuous first derivatives')
@click.pass_context
def recovery(ctx, **kwargs):
    vlog(ctx, 'Starting recovery')
    pushChain(ctx, 'recovery', **kwargs)
    data = ctx.obj['data']

    if kwargs['basistype'] is not None:
        if kwargs['basistype'] == 'ms' or kwargs['basistype'] == 'ns':
            basisType = 'serendipity'
        elif kwargs['basistype'] == 'mo':
            basisType = 'maximal-order'
    else:
        basisType = None
    #end

    for dat in data.iterator(kwargs['tag']):
        dg = GInterpModal(dat,
                          kwatgs['polyorder'], basisType, 
                          kwargs['interp'], kwargs['periodic'])
        numNodes = dg.numNodes
        numComps = int(ctx.obj['dataSets'][s].getNumComps() / numNodes)

        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        #dg.recovery(tuple(range(numComps)), stack=True)
        dg.recovery(0, inputs['c1'], stack=True)
    #end
    vlog(ctx, 'Finishing recovery')
#end
