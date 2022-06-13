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

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              help='Tag for the result')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.option('-x', type=click.STRING,
              help="Select components that will became the grid of the new dataset.")
@click.option('-y', type=click.STRING,
              help="Select components that will became the values of the new dataset.")
@click.pass_context
def val2coord(ctx, **kwargs):
    """Given a dataset (typically a DynVector) selects columns from it to
    create new datasets. For example, you can choose say column 1 to
    be the X-axis of the new dataset and column 2 to be the
    Y-axis. Multiple columns can be choosen using range specifiers and
    as many datasets are then created.

    """
    vlog(ctx, 'Starting val2coord')
    pushChain(ctx, 'val2coord', **kwargs)
    data = ctx.obj['data']

    activeSets = []
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    tags = list(data.tagIterator())
    outTag = kwargs['tag']
    if outTag is None:
        if len(tags) == 1:
            outTag = tags[0]
        else:
            outTag = 'val2coord'
        #end
    #end
    
    for setIdx, dat in data.iterator(kwargs['use'], enum=True):
        values = dat.getValues()
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

            x = values[..., xc]
            y = values[..., yc, np.newaxis]

            out = Data(tag=outTag,
                       label=kwargs['label'],
                       comp_grid=ctx.obj['compgrid'],
                       meta=dat.meta)
            out.push([x], y)
            out.color = 'C0'
            data.add(out)
        #end
        dat.deactivate()
    #end
    vlog(ctx, 'Finishing val2coord')
#end
