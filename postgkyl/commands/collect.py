import click
import numpy as np

from postgkyl.data import Data
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('-s', '--sumdata',
              is_flag=True,
              help="Sum data in the collected datasets (retain components)")
@click.option('-p', '--period',
              type=click.FLOAT,
              help="Specify a period to create epoch data instead of time data")
@click.option('-o', '--offset',
              default=0.0, type=click.FLOAT, show_default=True,
              help="Specify an offset to create epoch data instead of time data")
@click.option('-c', '--chunk', type=click.INT,
              help="Collect into chunks with specified length rather than into a single dataset")
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--outtag', '-o', default='collect',
              help='Specify a \'tag\' for the result.')
@click.option('--label', '-l', default='collect',
              help="Specify the custom label for the result.")
@click.pass_context
def collect(ctx, **kwargs):
    """Collect data from the active datasets and create a new combined
    dataset. The time-stamp in each of the active datasets is
    collected and used as the new X-axis. Data can be collected in
    chunks, in which case several datasets are created, each with the
    chunk-sized pieces collected into each new dataset.
    """
    vlog(ctx, 'Starting collect')
    pushChain(ctx, 'collect', **kwargs)
    data = ctx.obj['data']
    
    if kwargs['tag']:
        tagIterator = kwargs['tag'].split(',')
    else:
        tagIterator = data.tagIterator()
    #end
    for tag in tagIterator:
        time = [[]]
        values = [[]]
        grid = [[]]
        cnt = 0
        
        for i, dat in data.iterator(tag, enum=True):
            cnt += 1
            if kwargs['chunk'] and cnt > kwargs['chunk']:
                cnt = 1
                time.append([])
                values.append([])
                grid.append([])
            #end
            if dat.meta['time']:
                time[-1].append(dat.meta['time'])
            elif dat.meta['frame']:
                time[-1].append(dat.meta['frame'])
            else:
                time[-1].append(i)
            #end
            val = dat.getValues()
            if kwargs['sumdata']:
                numDims = dat.getNumDims()
                axis = tuple(range(numDims))
                values[-1].append(np.nansum(val, axis=axis))
            else:
                values[-1].append(val)
            #end
            if not grid[-1]:
                grid[-1] = dat.getGrid().copy()
            #end
        #end

        if cnt == 0:
            continue
        #end
        
        for i in range(len(time)):
            time[i] = np.array(time[i])
            values[i] = np.array(values[i])

            if kwargs['period'] is not None:
                time[i] = (time[i] - kwargs['offset']) % kwargs['period']
            #end

            sortIdx = np.argsort(time[i])
            time[i] = time[i][sortIdx]
            values[i] = values[i][sortIdx]

            if kwargs['sumdata']:
                grid[i] = [time[i]]
            else:
                grid[i].insert(0, np.array(time[i]))
            #end

            #vlog(ctx, 'collect: Creating {:d}D data with shape {}'.format(len(grid), values[i].shape))
            out = Data(tag=kwargs['outtag'],
                       stack=ctx.obj['stack'],
                       compgrid=ctx.obj['compgrid'],
                       label=kwargs['label'])
            out.push(values[i], grid[i])
            data.add(out)
        #end

        data.deactivateAll(tag)
    #end

    vlog(ctx, 'Finishing collect')
#end
