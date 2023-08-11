import click
import numpy as np

from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.commands.util import verb_print
from postgkyl.data import GData

@click.command(help='Interpolate DG data onto a uniform mesh.')
@click.option('--basistype', '-b',
              type=click.Choice(['ms', 'ns', 'mo', 'mt', 'gkhyb', 'pkpmhyb']),
              help='Specify DG basis.')
@click.option('--polyorder', '-p', type=click.INT,
              help='Specify polynomial order.')
@click.option('--interp', '-i', type=click.INT,
              help='Interpolation onto a general mesh of specified amount.')
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.option('--read', '-r', type=click.BOOL,
              help='Read from general interpolation file.')
@click.pass_context
def interpolate(ctx, **kwargs):
  verb_print(ctx, 'Starting interpolate')
  data = ctx.obj['data']

  basisType = None
  isModal = None
  if kwargs['basistype'] is not None:
    if kwargs['basistype'] == 'ms':
      basisType = 'serendipity'
      isModal = True
    elif kwargs['basistype'] == 'ns':
      basisType = 'serendipity'
      isModal = False
    elif kwargs['basistype'] == 'mo':
      basisType = 'maximal-order'
      isModal = True
    elif kwargs['basistype'] == 'mt':
      basisType = 'tensor'
      isModal = True
    elif kwargs['basistype'] == 'gkhyb':
      basisType = 'gkhybrid'
      isModal = True
    elif kwargs['basistype'] == 'pkpmhyb':
      basisType = 'pkpmhybrid'
      isModal = True
    #end
  #end

  for dat in data.iterator(kwargs['use']):
    if kwargs['basistype'] is None and dat.ctx['basisType'] is None:
      ctx.fail(click.style("ERROR in interpolate: no 'basistype' was specified and dataset {:s} does not have required ctxdata".format(dat.getLabel()), fg='red'))
    #end

    if isModal or dat.ctx['isModal']:
      dg = GInterpModal(dat,
                        kwargs['polyorder'], kwargs['basistype'],
                        kwargs['interp'], kwargs['read'])
    else:
      dg = GInterpNodal(dat,
                        kwargs['polyorder'], basisType,
                        kwargs['interp'], kwargs['read'])
    #end

    numNodes = dg.numNodes
    numComps = int(dat.getNumComps() / numNodes)

    if kwargs['tag']:
      out = GData(tag=kwargs['tag'],
                  label=kwargs['label'],
                  comp_grid=ctx.obj['compgrid'],
                  ctx=dat.ctx)
      grid, values = dg.interpolate(tuple(range(numComps)))
      out.push(grid, values)
      data.add(out)
    else:
      dg.interpolate(tuple(range(numComps)), overwrite=True)
    #end
  #end
  verb_print(ctx, 'Finishing interpolate')
#end
