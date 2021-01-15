import click
import numpy as np

from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.commands.util import vlog, pushChain

from postgkyl.modalDG import interpolate as interpFn

@click.command(help='Interpolate DG data onto a uniform mesh.')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo', 'mt']),
              help='Specify DG basis.')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order.')
@click.option('--interp', '-i', type=click.INT,
              help='Interpolation onto a general mesh of specified amount.')
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--read', '-r', type=click.BOOL,
              help='Read from general interpolation file.')
@click.option('-n', '--new', is_flag=True,
              help="for testing purposes")
@click.pass_context
def interpolate(ctx, **kwargs):
    vlog(ctx, 'Starting interpolate')
    pushChain(ctx, 'interpolate', **kwargs)

    basisType = None
    isModal = None
    if kwargs['basistype'] is not None:
        if kwargs['basistype'] == 'ms':
            basisType = 'serendipity'
            isModal = True
        elif kwargs['basistype'] == 'ns':
            basisType = 'serendipity'
            isModal = False
        elif kwargs['basistype'] == 'mo':
            basisType = 'maximal-order'
            isModal = True
        elif kwargs['basistype'] == 'mt':
            basisType = 'tensor'
            isModal = True
        #end
    #end
    
    for dat in ctx.obj['data'].iterator(kwargs['tag']):
        if kwargs['basistype'] is None and dat.basisType is None:
            click.echo(click.style("ERROR in interpolate: no 'basistype' was specified and dataset {:s} does not have required metadata".format(dat.getLabel()), fg='red'))
            ctx.exit()
        #end
        
        if isModal or dat.isModal:
            dg = GInterpModal(dat,
                              kwargs['polyorder'], kwargs['basistype'], 
                              kwargs['interp'], kwargs['read'])
        else:
            dg = GInterpNodal(dat,
                              kwargs['polyorder'], basisType,
                              kwargs['interp'], kwargs['read'])
        #end
            
        numNodes = dg.numNodes
        numComps = int(dat.getNumComps() / numNodes)
        
        if not kwargs['new']:
            dg.interpolate(tuple(range(numComps)), stack=True)
        else:
            interpFn(dat, kwargs['polyorder'])
        #end
    #end
    vlog(ctx, 'Finishing interpolate')
#end
