import click
import numpy as np

from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Interpolate DG data on a uniform mesh')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Interpolation onto a general mesh of specified amount')
@click.option('--read', '-r', type=click.BOOL,
              help='Read from general interpolation file')
@click.pass_context
def interpolate(ctx, **inputs):
    vlog(ctx, 'Starting interpolate')
    pushChain(ctx, 'interpolate', **inputs)

    if inputs['basistype'] is not None:
        if inputs['basistype'] == 'ms' or inputs['basistype'] == 'ns':
            basisType = 'serendipity'
        elif inputs['basistype'] == 'mo':
            basisType = 'maximal-order'
    else:
        basisType = None
    #end
    
    for s in ctx.obj['sets']:
        if ctx.obj['dataSets'][s].modal or inputs['basistype'] == 'ms' or inputs['basistype'] == 'mo':
            dg = GInterpModal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], basisType, 
                              inputs['interp'], inputs['read'])
        else:
            dg = GInterpNodal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], basisType,
                              inputs['interp'], inputs['read'])
        #end
        numNodes = dg.numNodes
        numComps = int(ctx.obj['dataSets'][s].getNumComps() / numNodes)

        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        dg.interpolate(tuple(range(numComps)), stack=True)
    #end
    vlog(ctx, 'Finishing interpolate')
#end
