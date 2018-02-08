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
    pushChain(ctx, 'dg.interpolate', **inputs)

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
        numComps = ctx.obj['dataSets'][s].getNumComps() / numNodes
        numDims = ctx.obj['dataSets'][s].getNumDims()

        vlog(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
        grid, values = dg.interpolate(0)

        if numComps > 1:
            for comp in range(1, numComps):
                grid, tmp = dg.interpolate(comp)
                values = np.append(values, tmp, axis=numDims)
 
        #label = 'proj_{:s}_{:d}'.format(inputs['basis'], inputs['polyorder'])
        ctx.obj['dataSets'][s].pushGrid(grid)
        ctx.obj['dataSets'][s].pushValues(values)
    vlog(ctx, 'Finishing interpolate')

@click.command(help='Calculate the derivative of DG data on a uniform mesh')
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
def derivative(ctx, **inputs):
    pass
