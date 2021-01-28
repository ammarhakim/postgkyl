import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain

#---------------------------------------------------------------------
#-- Helper functions -------------------------------------------------

def _getIterableIdx(idx, length):
    def _int(i, length):
        i = int(i)
        if i >= 0:
            return i
        else:
            return int(length + i)
        #end
    #end
    
    if idx is None:
        return range(length)
    elif ',' in idx:
        s = idx.split(',')
        return [_int(i, length) for i in s]
    elif ':' in idx:
        s = idx.split(':')
        si = [0, length, 1]
        if s[0]:
            si[0] = _int(s[0], length)
        if s[1]:
            si[1] = _int(s[1], length)
        if len(s) > 2:
            si[2] = int(s[2])
        return range(si[0], si[1], si[2])
    else:
        return [_int(idx, length)]
    #end
#end

#---------------------------------------------------------------------
#-- Main functions ---------------------------------------------------

@click.command()
@click.option('--tag', '-t', type=click.STRING,
              help='Tag(s) to apply to (comma-separated)')
@click.option('--index', '-i', type=click.STRING,
              help="Dataset indices (e.g., '1', '0,2,5', or '1:6:2')")
@click.option('--focused', '-f', is_flag=True,
              help='Leave unspecified datasets untouched')
@click.pass_context
def activate(ctx, **kwargs):
    """Select datasets(s) to pass further down the command
    chain. Datasets are indexed starting 0. Multiple datasets can be
    selected using a comma separated list or a range specifier. Unless
    '--focused' is selected, all unselected datasets will be
    deactivated.

    '--tag' and '--index' allow to specify tags and indices. The not
    specified, 'activate' applies to all. Both parameters support
    comma-separated values. '--index' also supports slices following
    the Python conventions, e.g., '3:7' or ':-5:2'.

    'info' command (especially with the '-ac' flags) can be helpful
    when activating/deactivating multiple datasets.

    """
    vlog(ctx, 'Starting activate')
    pushChain(ctx, 'activate', **kwargs)
    data = ctx.obj['data']

    if not kwargs['focused']:
        data.deactivateAll()
    #end

    for tag in data.tagIterator(kwargs['tag']):
        numFiles = data.getNumDatasets(tag=tag, onlyActive=False)
        for i, dat in data.iterator(tag=tag, enum=True, onlyActive=False):
            if i in _getIterableIdx(kwargs['index'], numFiles):
                dat.activate()
            #end
        #end
    #end
    vlog(ctx, 'Finishing activate')
#end

@click.command()
@click.option('--tag', '-t', type=click.STRING,
              help='Tag(s) to apply to (comma-separated)')
@click.option('--index', '-i', type=click.STRING,
              help="Dataset indices (e.g., '1', '0,2,5', or '1:6:2')")
@click.option('--focused', '-f', is_flag=True,
              help='Leave unspecified datasets untouched')
@click.pass_context
def deactivate(ctx, **kwargs):
    """Select datasets(s) to pass further down the command
    chain. Datasets are indexed starting 0. Multiple datasets can be
    selected using a comma separated list or a range specifier. Unless
    '--focused' is selected, all unselected datasets will be
    activated.

    '--tag' and '--index' allow to specify tags and indices. The not
    specified, 'deactivate' applies to all. Both parameters support
    comma-separated values. '--index' also supports slices following
    the Python conventions, e.g., '3:7' or ':-5:2'.

    'info' command (especially with the '-ac' flags) can be helpful
    when activating/deactivating multiple datasets.

    """
    vlog(ctx, 'Starting deactivate')
    pushChain(ctx, 'deactivate', **kwargs)
    data = ctx.obj['data']

    if kwargs['focused']:
        data.activateAll()
    #end

    for tag in data.tagIterator(kwargs['tag']):
        numFiles = data.getNumDatasets(tag=tag, onlyActive=False)
        for i, dat in data.iterator(tag=tag, enum=True, onlyActive=False):
            if i in _getIterableIdx(kwargs['index'], numFiles):
                dat.deactivate()
            #end
        #end
    #end
    vlog(ctx, 'Finishing deactivate')
#end
