import click
import numpy as np

from postgkyl.data import GInterpModal
from postgkyl.commands.util import verb_print
from postgkyl.data import GData

@click.command(help='Interpolate DG data on a uniform mesh')
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo']),
              help='Specify DG basis')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order')
@click.option('--interp', '-i', type=click.INT,
              help='Number of poins to evaluate on')
@click.option('-r', '--periodic', is_flag=True,
              help='Flag for periodic boundary conditions')
@click.option('-c', '--c1', is_flag=True,
              help='Enforce continuous first derivatives')
@click.pass_context
def recovery(ctx, **kwargs):
  verb_print(ctx, 'Starting recovery')
  data = ctx.obj['data']

  if kwargs['basistype'] is not None:
    if kwargs['basistype'] == 'ms' or kwargs['basistype'] == 'ns':
      basisType = 'serendipity'
    elif kwargs['basistype'] == 'mo':
      basisType = 'maximal-order'
    #end
  else:
    basisType = None
  #end

  for dat in data.iterator(kwargs['use']):
    dg = GInterpModal(dat,
                      kwargs['polyorder'], basisType,
                      kwargs['interp'], kwargs['periodic'])
    numNodes = dg.numNodes
    numComps = int(dat.getNumComps() / numNodes)

    #verb_print(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
    #dg.recovery(tuple(range(numComps)), stack=True)
    if kwargs['tag']:
      out = GData(tag=kwargs['tag'],
                  label=kwargs['label'],
                  comp_grid=ctx.obj['compgrid'],
                  ctx=dat.ctx)
      grid, values = dg.recovery(0, kwargs['c1'])
      out.push(grid, values)
      data.add(out)
    else:
      dg.recovery(0, kwargs['c1'], overwrite=True)
    #end
  #end
  verb_print(ctx, 'Finishing recovery')
#end
