import click
from glob import glob
from os import path
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('--varname', '-d', multiple=True, default=['CartGridField'],
              help="Allows to specify the Adios variable name (default is 'CartGridField')")
@click.pass_context
def load(ctx, **kwargs):
    vlog(ctx, 'Starting load')
    pushChain(ctx, 'load', **kwargs)

    idx = ctx.obj['inDataStringsLoaded']
    inDataString = ctx.obj['inDataStrings'][idx]
    
    if "*" in inDataString or "?" in inDataString or "!"  in inDataString:
        files = glob(str(inDataString))
        files = [f for f in files if f.find("restart") < 0]
        def _crush(s): # Temp function used as a sorting key
            splitted = s.split('_')
            tmp = splitted[-1].split('.')
            splitted[-1] = int(tmp[0])
            splitted.append(tmp[1])
            return tuple(splitted)
        try:
            files = sorted(files, key=_crush)
        except Exception:
            click.echo(click.style("WARNING: The loaded files appear to be of different types. Sorting is turned off.", fg='yellow'))
        #end
    else:
         files = [inDataString]
    #end
    print(files)

    cnt = len(ctx.obj['dataSets'])
    for var in kwargs['varname']:
        for fn in files:
            try:
                ctx.obj['dataSets'].append(Data(fn, varName=var))
                ctx.obj['setIds'].append(cnt)
                cnt = cnt + 1
            except NameError:
                click.fail(click.style("ERROR: File(s) '{:s}' not found or empty".format(fn), fg='red'))
            #end
        #end
    #end
    ctx.obj['sets'] = range(cnt)

    ctx.obj['inDataStringsLoaded'] = ctx.obj['inDataStringsLoaded'] + 1
    vlog(ctx, 'Finishing load')
#end
