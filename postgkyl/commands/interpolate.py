import click
import numpy as np

from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.commands.util import vlog, pushChain

from postgkyl.modalDG import interpolate as interpFn

@click.command(help='Interpolate DG data onto a uniform mesh.')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo', 'mt']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Interpolation onto a general mesh of specified amount')
@click.option('--read', '-r', type=click.BOOL,
              help='Read from general interpolation file')
@click.option('-n', '--new', is_flag=True,
              help="for testing purposes")
@click.pass_context
def interpolate(ctx, **inputs):
    vlog(ctx, 'Starting interpolate')
    pushChain(ctx, 'interpolate', **inputs)

    basisType = None
    isModal = None
    if inputs['basistype'] is not None:
        if inputs['basistype'] == 'ms':
            basisType = 'serendipity'
            isModal = True
        elif inputs['basistype'] == 'ns':
            basisType = 'serendipity'
            isModal = False
        elif inputs['basistype'] == 'mo':
            basisType = 'maximal-order'
            isModal = True
        elif inputs['basistype'] == 'mt':
            basisType = 'tensor'
            isModal = True
        #end
    #end
    
    for i, s in enumerate(ctx.obj['sets']):
        if inputs['basistype'] is None and ctx.obj['dataSets'][s].basisType is None:
            click.echo(click.style("ERROR in interpolate: no 'basistype' was specified and dataset {:d} does not have required metadata".format(i), fg='red'))
            ctx.exit()
        #end
        
        if isModal or ctx.obj['dataSets'][s].isModal:
            dg = GInterpModal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], inputs['basistype'], 
                              inputs['interp'], inputs['read'])
        else:
            dg = GInterpNodal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], basisType,
                              inputs['interp'], inputs['read'])
        #end
            
        numNodes = dg.numNodes
        numComps = int(ctx.obj['dataSets'][s].getNumComps() / numNodes)
        
        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        if not inputs['new']:
            dg.interpolate(tuple(range(numComps)), stack=True)
        else:
            interpFn(ctx.obj['dataSets'][s], inputs['polyorder'])
        #end
    #end
    vlog(ctx, 'Finishing interpolate')
#end
