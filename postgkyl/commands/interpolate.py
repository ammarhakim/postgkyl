import click
import numpy as np

from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Interpolate DG data on a uniform mesh')
@click.option('--basis', '-b', prompt=True,
              type=click.Choice(['ns', 'ms', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', prompt=True, type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Interpolation onto a general mesh of specified amount')
@click.option('--read', '-r', type=click.BOOL,
              help='Read from general interpolation file')
@click.pass_context
def interpolate(ctx, **inputs):
    vlog(ctx, 'Starting interpolate')
    pushChain(ctx, 'interpolate', **inputs)

    for s in ctx.obj['sets']:
        if inputs['basis'] == 'ms' or inputs['basis'] == 'mo':
            dg = GInterpModal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], inputs['basis'], 
                              inputs['interp'], inputs['read'])
        elif inputs['basis'] == 'ns':
            dg = GInterpNodal(ctx.obj['dataSets'][s],
                              inputs['polyorder'], inputs['basis'],
                              inputs['interp'], inputs['read'])
        numNodes = dg.numNodes
        numComps = int(ctx.obj['dataSets'][s].getNumComps() / numNodes)
        numDims = ctx.obj['dataSets'][s].getNumDims()

        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        dg.interpolate(tuple(range(numComps)), stack=True)
    vlog(ctx, 'Finishing interpolate')
