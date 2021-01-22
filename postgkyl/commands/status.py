import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain



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


@click.command()
@click.option('--tag', '-t',
              help='Specify the tag(s) to apply to (default all tags).')
@click.option('--idx', '-i', type=click.STRING,
              help='Dataset indices')
@click.option('--exclusive', '-e', is_flag=True,
              help='Deactivate unspecified datasets')
@click.pass_context
def activate(ctx, **kwargs):
    """Select datasets(s) to pass further down the command chain. Datasets
    are indexed starting 0. Multiple datasets can be selected using a
    comma separated list or a range specifier.

    When '--tag' is specified, the sellection is applied only on the
    set tags (comma separated list is acceptable). When '--idx' is not
    used, all datasets are activated.

    When neither '--tag' nor '--idx' are specified, all datasets are
    activated.

    """
    vlog(ctx, 'Starting activate')
    pushChain(ctx, 'activate', **kwargs)
    data = ctx.obj['data']

    if kwargs['exclusive']:
        data.deactivateAll()
    #end

    if kwargs['tag']:
        tagIterator = kwargs['tag'].split(',')
    else:
        tagIterator = data.tagIterator()
    #end
    for tag in tagIterator:
        numFiles = data.getNumDatasets(tag=tag, onlyActive=False)
        for i, dat in data.iterator(tag=tag, enum=True, onlyActive=False):
            if i in _getIterableIdx(kwargs['idx'], numFiles):
                dat.activate()
            #end
        #end
    #end
    vlog(ctx, 'Finishing activate')
#end

@click.command()
@click.option('--tag', '-t',
              help='Specify the tag(s) to apply to (default all tags).')
@click.option('--idx', '-i', type=click.STRING,
              help='Dataset indices')
@click.option('--exclusive', '-e', is_flag=True,
              help='Activate unspecified datasets')
@click.pass_context
def deactivate(ctx, **kwargs):
    """Select datasets(s) to pass further down the command chain. Datasets
    are indexed starting 0. Multiple datasets can be selected using a
    comma separated list or a range specifier.

    When '--tag' is specified, the sellection is applied only on the
    set tags (comma separated list is acceptable). When '--idx' is not
    used, all datasets are deactivated.

    When neither '--tag' nor '--idx' are specified, all datasets are
    deactivated.

    """
    vlog(ctx, 'Starting deactivate')
    pushChain(ctx, 'deactivate', **kwargs)
    data = ctx.obj['data']

    if kwargs['exclusive']:
        data.activateAll()
    #end

    if kwargs['tag']:
        tagIterator = kwargs['tag'].split(',')
    else:
        tagIterator = data.tagIterator()
    #end
    for tag in tagIterator:
        numFiles = data.getNumDatasets(tag=tag, onlyActive=False)
        for i, dat in data.iterator(tag=tag, enum=True, onlyActive=False):
            if i in _getIterableIdx(kwargs['idx'], numFiles):
                dat.deactivate()
            #end
        #end
    #end
    vlog(ctx, 'Finishing deactivate')
#end
