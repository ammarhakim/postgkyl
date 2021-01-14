import click
from glob import glob
from os import path
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('--z0', help="Partial file load: 0th coord (either int or slice)")
@click.option('--z1', help="Partial file load: 1st coord (either int or slice)")
@click.option('--z2', help="Partial file load: 2nd coord (either int or slice)")
@click.option('--z3', help="Partial file load: 3rd coord (either int or slice)")
@click.option('--z4', help="Partial file load: 4th coord (either int or slice)")
@click.option('--z5', help="Partial file load: 5th coord (either int or slice)")
@click.option('--component', '-c',
              help="Partial file load: comps (either int or slice)")
@click.option('--tag', '-t', default="default",
              help="Specily tag for data (default: \"default\")")
@click.option('--compgrid', is_flag=True,
              help="Disregard the mapped grid information")
@click.option('--varname', '-d', multiple=True,
              help="Allows to specify the Adios variable name (default is 'CartGridField')")
@click.pass_context
def load(ctx, **kwargs):
    vlog(ctx, 'Starting load')
    pushChain(ctx, 'load', **kwargs)

    idx = ctx.obj['inDataStringsLoaded']
    inDataString = ctx.obj['inDataStrings'][idx]

    # Handling the wildcard characters
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

    # Resolve the local/global variable names and partial loading
    # The local settings take a precedents but a warning is going to appear
    z0 = _pickCut(ctx, kwargs, 0)
    z1 = _pickCut(ctx, kwargs, 1)
    z2 = _pickCut(ctx, kwargs, 2)
    z3 = _pickCut(ctx, kwargs, 3)
    z4 = _pickCut(ctx, kwargs, 4)
    z5 = _pickCut(ctx, kwargs, 5)
    comp = _pickCut(ctx, kwargs, 6)
    if kwargs['varname'] and ctx.obj['globalVarNames']:
        varNames = kwargs['varname']
        click.echo(click.style("WARNING: The local 'varname' is overwriting the global 'varname'", fg='yellow'))
    elif kwargs['varname']:
        varNames = kwargs['varname']
    elif ctx.obj['globalVarNames']:
        varNames = ctx.obj['globalVarNames']
    else:
        varNames = ['CartGridField']
    #end
    
    for var in varNames:
        for fn in files:
            try:
                ctx.obj['data'].add(Data(fileName=fn, tag=kwargs['tag'],
                                         stack=ctx.obj['stack'],
                                         compgrid=ctx.obj['compgrid'],
                                         z0=z0, z1=z1, z2=z2,
                                         z3=z3, z4=z4, z5=z5,
                                         comp=comp, varName=var))
            except NameError:
                click.fail(click.style("ERROR: File(s) '{:s}' not found or empty".format(fn), fg='red'))
            #end
        #end
    #end

    ctx.obj['inDataStringsLoaded'] = ctx.obj['inDataStringsLoaded'] + 1
    vlog(ctx, 'Finishing load')
#end

def _pickCut(ctx, kwargs, zn):
    nm = 'z{:d}'.format(zn)
    if zn == 6: # This little hack allows to apply the same function for components as well
        nm = 'component'
    #end
    if kwargs[nm] and ctx.obj['globalCuts'][zn]:
        click.echo(click.style("WARNING: The local '{:s}' is overwriting the global '{:s}'".format(nm, nm),
                               fg='yellow'))
        return kwargs[nm]
    elif kwargs[nm]:
        return kwargs[nm]
    elif ctx.obj['globalCuts'][zn]:
        return ctx.obj['globalCuts'][zn]
    else:
        return None
    #end
#end
