import click
import numpy as np

from postgkyl.data import Data
from postgkyl.commands.util import vlog, pushChain

def _getRange(strIn, length):
    if len(strIn.split(',')) > 1:
        return np.array(strIn.split(','), np.int)
    elif strIn.find(':') >= 0:
        strSplit = strIn.split(':')
        
        if strSplit[0] == '':
            sIdx = 0
        else:
            sIdx = int(strSplit[0])
            if sIdx < 0:
                sIdx = length+sIdx
            #end
        #end
        
        if strSplit[1] == '':
            eIdx = length
        else:
            eIdx = int(strSplit[1])
            if eIdx < 0:
                eIdx = length+eIdx
            #end
        #end

        inc = 1
        if len(strSplit) > 2 and strSplit[2] != '':
            inc = int(strSplit[2])
        #end
        return np.arange(sIdx, eIdx, inc)
    else:
        return np.array([int(strIn)])
    #end
#end

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
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for setIdx, s in enumerate(ctx.obj['sets']):
        values = ctx.obj['dataSets'][s].getValues()
        xComps = _getRange(kwargs['x'], len(values[0, :]))
        yComps = _getRange(kwargs['y'], len(values[0, :]))

        if len(xComps) > 1 and len(xComps) != len(yComps):
            click.echo(click.style("ERROR 'val2coord': Length of the x-components ({:d}) is greater than 1 and not equal to the y-components ({:d}).".format(len(xComps), len(yComps)), fg='red'))
            ctx.exit()
        #end
        
        for i, yc in enumerate(yComps):
            if len(xComps) > 1:
                xc = xComps[i]
            else:
                xc = xComps[0]
            #end
            
            values = ctx.obj['dataSets'][s].getValues()

            x = values[..., xc]
            y = values[..., yc, np.newaxis]

            newSetIdx = len(ctx.obj['dataSets'])
            ctx.obj['setIds'].append(newSetIdx)
            ctx.obj['dataSets'].append(Data())
            ctx.obj['labels'].append('val2coord_{:d}'.format(i))
            ctx.obj['dataSets'][newSetIdx].pushGrid([x])
            ctx.obj['dataSets'][newSetIdx].pushValues(y)
            ctx.obj['dataSets'][newSetIdx].time = None
            ctx.obj['dataSets'][newSetIdx].fileName = None
            ctx.obj['dataSets'][newSetIdx].color = colors[setIdx]
            vlog(ctx, 'val2coord: activated data set #{:d}'.format(newSetIdx))
            activeSets.append(newSetIdx)
        #end
    #end
    ctx.obj['sets'] = activeSets
    vlog(ctx, 'Finishing val2coord')
#end
