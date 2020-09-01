import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Creates new dataset(s) from values of curent dataset(s), i.e., turns some values into grid')
@click.option('-x', type=click.STRING,
              help="Select components that will became the grid of the new dataset.")
@click.option('-y', type=click.STRING,
              help="Select components that will became the values of the new dataset.")
@click.pass_context
def val2coord(ctx, **kwargs):
    vlog(ctx, 'Starting select')
    pushChain(ctx, 'val2coord', **kwargs)

    activeSets = []

    xComps = kwargs['x'].split(',')
    yComps = kwargs['y'].split(',')
    if len(xComps) > 1 and len(xComps) != len(yComps):
        click.echo(click.style("ERROR 'val2coord': Length of the x-components ({:d}) is greater than 1 and not equal to the y-components ({:d}).".format(len(xComps), len(yComps)), fg='red'))
        ctx.exit()
    #end
    
    for s in ctx.obj['sets']:
        for i, yc in enumerate(yComps):
            yc = int(yc)
            if len(xComps) > 1:
                xc = int(xComps[i])
            else:
                xc = int(xComps[0])
            #end
            
            values = ctx.obj['dataSets'][s].getValues()

            x = values[..., xc]
            y = values[..., yc, np.newaxis]

            idx = len(ctx.obj['dataSets'])
            ctx.obj['setIds'].append(idx)
            ctx.obj['dataSets'].append(GData())
            ctx.obj['labels'].append(st)
            ctx.obj['dataSets'][idx].pushGrid([x])
            ctx.obj['dataSets'][idx].pushValues(y)
            ctx.obj['dataSets'][idx].time = None
            ctx.obj['dataSets'][idx].fileName = None
            vlog(ctx, 'val2coord: activated data set #{:d}'.format(idx))
            activeSets.append(idx)
        #end
    #end
    ctx.obj['sets'] = activeSets
    vlog(ctx, 'Finishing val2coord')
#end
