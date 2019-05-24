import click
import numpy as np

from postgkyl.data import GInterpModal
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Interpolate DG data on a uniform mesh')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Number of poins to evaluate on')
@click.option('-r', '--periodic', is_flag=True,
              help='Flag for periodic boundary conditions')
@click.pass_context
def recovery(ctx, **inputs):
    vlog(ctx, 'Starting recovery')
    pushChain(ctx, 'recovery', **inputs)

    if inputs['basistype'] is not None:
        if inputs['basistype'] == 'ms' or inputs['basistype'] == 'ns':
            basisType = 'serendipity'
        elif inputs['basistype'] == 'mo':
            basisType = 'maximal-order'
    else:
        basisType = None
    #end

    for s in ctx.obj['sets']:
        dg = GInterpModal(ctx.obj['dataSets'][s],
                          inputs['polyorder'], basisType, 
                          inputs['interp'], inputs['periodic'])
        numNodes = dg.numNodes
        numComps = int(ctx.obj['dataSets'][s].getNumComps() / numNodes)

        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        #dg.recovery(tuple(range(numComps)), stack=True)
        dg.recovery(0, stack=True)
    #end
    vlog(ctx, 'Finishing recovery')
#end
