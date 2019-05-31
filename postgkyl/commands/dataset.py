import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Select data sets(s)')
@click.option('-i', '--idx', type=click.STRING,
              help='Data set indices')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def dataset(ctx, **kwargs):
    idx = kwargs['idx']
    if kwargs['allsets']:
        vlog(ctx, 'Selecting all datasets')
    else:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
    #end
    pushChain(ctx, 'dataset', **kwargs)

    if kwargs['allsets'] is False:
        vlog(ctx, 'Selecting data set(s): {:s}'.format(idx))
        if len(idx.split(',')) > 1:
            sets = idx.split(',')
            ctx.obj['sets'] = [int(s) for s in sets]            
        elif len(idx.split(':')) == 2:
            sets = idx.split(':')
            if sets[0] == '':
               sets[0] = 0
            #end
            if sets[1] == '':
               sets[1] = len(ctx.obj['dataSets'])
            #end
            if int(sets[1]) < 0:
               sets[1] = len(ctx.obj['dataSets']) + int(sets[1]) + 1
            #end
            ctx.obj['sets'] = range(int(sets[0]), int(sets[1]))
        else:
            ctx.obj['sets'] = [int(idx)]
        #end
    else:
        vlog(ctx, 'Selecting all data sets'.format(idx))
        ctx.obj['sets'] = range(len(ctx.obj['dataSets']))
    #end
