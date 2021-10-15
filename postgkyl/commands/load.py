import click
import glob
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
from postgkyl.data import GInterpModal

def _pickCut(ctx, kwargs, zn):
  nm = 'z{:d}'.format(zn)
  if zn == 6: # This little hack allows to apply the same function for
              # components as well
    nm = 'component'
  #end
  if kwargs[nm] and ctx.obj['globalCuts'][zn]:
    click.echo(click.style(
      'WARNING: The local \'{:s}\' is overwriting the global \'{:s}\''.format(nm, nm),
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

def _crush(s): # Temp function used as a sorting key
  splitted = s.split('_')
  tmp = splitted[-1].split('.')
  splitted[-1] = int(tmp[0])
  splitted.append(tmp[1])
  return tuple(splitted)
#end

@click.command(hidden=True)
@click.option('--z0', help='Partial file load: 0th coord (either int or slice)')
@click.option('--z1', help='Partial file load: 1st coord (either int or slice)')
@click.option('--z2', help='Partial file load: 2nd coord (either int or slice)')
@click.option('--z3', help='Partial file load: 3rd coord (either int or slice)')
@click.option('--z4', help='Partial file load: 4th coord (either int or slice)')
@click.option('--z5', help='Partial file load: 5th coord (either int or slice)')
@click.option('--component', '-c',
              help='Partial file load: comps (either int or slice)')
@click.option('--tag', '-t', default='default',
              help='Specily tag for data (default: \'default\')')
@click.option('--compgrid', is_flag=True,
              help='Disregard the mapped grid information')
@click.option('--varname', '-d', multiple=True,
              help='Allows to specify the Adios variable name (default is \'CartGridField\')')
@click.option('--label', '-l',
              help='Allows to specify the custom label')
@click.option('--c2p', type=click.STRING,
              help='Specify the file name containing c2p mapped coordinates')
@click.option('--fv', is_flag=True,
              help='Tag finite volume data when using c2p mapped coordinates')
@click.pass_context
def load(ctx, **kwargs):
  vlog(ctx, 'Starting load')
  pushChain(ctx, 'load', **kwargs)
  data = ctx.obj['data']

  idx = ctx.obj['inDataStringsLoaded']
  inDataString = ctx.obj['inDataStrings'][idx]

  # Handling the wildcard characters
  if '*' in inDataString or '?' in inDataString or '!'  in inDataString:
    files = glob.glob(str(inDataString))
    files = [f for f in files if f.find('restart') < 0]
    try:
      files = sorted(files, key=_crush)
    except Exception:
      click.echo(click.style(
        'WARNING: The loaded files appear to be of different types. Sorting is turned off.',
        fg='yellow'))
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
    click.echo(click.style(
      'WARNING: The local \'varname\' is overwriting the global \'varname\'',
      fg='yellow'))
  elif kwargs['varname']:
    varNames = kwargs['varname']
  elif ctx.obj['globalVarNames']:
    varNames = ctx.obj['globalVarNames']
  else:
    varNames = ['CartGridField']
  #end

  if kwargs['c2p'] and ctx.obj['global_c2p']:
    mapc2p_name = kwargs['c2p']
    click.echo(click.style(
      'WARNING: The local \'c2p\' is overwriting the global \'c2p\'',
      fg='yellow'))
  elif kwargs['c2p']:
    mapc2p_name = kwargs['c2p']
  elif ctx.obj['global_c2p']:
    mapc2p_name = ctx.obj['global_c2p']
  else:
    mapc2p_name = None
  #end

  for var in varNames:
    for fn in files:
      try:
        dat = Data(file_name=fn, tag=kwargs['tag'],
                    comp_grid=ctx.obj['compgrid'],
                    z0=z0, z1=z1, z2=z2,
                    z3=z3, z4=z4, z5=z5,
                    comp=comp, var_name=var,
                    label = kwargs['label'],
                    mapc2p_name = mapc2p_name)
        
        if kwargs['fv']:
          dg = GInterpModal(dat, 0, 'ms')
          dg.interpolateGrid(overwrite=True)
        #end
        data.add(dat)
      except NameError:
        ctx.fail(click.style(
          'Failed to load the variable \'{:s}\' from \'{:s}\''.format(var, fn),
          fg='red'))
      #end
    #end
  #end

  data.setUniqueLabels()

  ctx.obj['inDataStringsLoaded'] += 1
  vlog(ctx, 'Finishing load')
#end
