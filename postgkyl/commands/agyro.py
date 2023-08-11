import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import verb_print
from postgkyl.tools import get_agyro, get_gkyl_10m_agyro


@click.command()
@click.option('--measure', '-m', default='frobenius', show_default=True,
              type=click.Choice(['swisdak', 'frobenius']),
              help='Specify how to calculate agyrotropy.')
@click.option('--pressure', '-p',
              default='pressure', show_default=True,
              help="Tag for input pressure.")
@click.option('--bfield', '-b',
              default='field', show_default=True,
              help="Tag for input EM field.")
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help='Custom label for the result')
@click.pass_context
def agyro(ctx, **kwargs):
  """Compute a measure of agyrotropy. Default measure is taken from
  Swisdak 2015. Optionally computes agyrotropy as Frobenius norm of
  agyrotropic pressure tensor.
  """
  verb_print(ctx, 'Starting agyro')

  data = ctx.obj['data'] # shortcut

  tag = 'agyro'
  if kwargs['tag']:
    tag = kwargs['tag']
  #end

  for pressure, bfield in zip(data.iterator(kwargs['pressure']),
                              data.iterator(kwargs['bfield'])):
    grid, agyro = get_agyro(p_data=pressure, b_data=bfield, measure=kwargs['measure'])
    out = GData(tag=tag,
                label=kwargs['label'],
                comp_grid=ctx.obj['compgrid'],
                ctx=pressure.ctx)
    out.push(grid, agyro)
    data.add(out)
  #end

  verb_print(ctx, 'Finishing agyro')
#end


@click.command()
@click.option('--measure', '-m', default='frobenius', show_default=True,
              type=click.Choice(['swidak', 'frobenius']),
              help='Specify how to calculate agyrotropy.')
@click.option('--species', '-s',
              help="Tag for input pressure.")
@click.option('--field', '-f',
              help="Tag for input EM field.")
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help='Custom label for the result')
@click.pass_context
def mom_agyro(ctx, **kwargs):
  """Compute a measure of agyrotropy. Default measure is taken from
  Swisdak 2015. Optionally computes agyrotropy as Frobenius norm of
  agyrotropic pressure tensor.
  """
  verb_print(ctx, 'Starting agyro')

  data = ctx.obj['data'] # shortcut

  tag = 'agyro'
  if kwargs['tag']:
    tag = kwargs['tag']
  #end

  for species, field in zip(data.iterator(kwargs['species']),
                            data.iterator(kwargs['field'])):
    grid, agyro = get_gkyl_10m_agyro(species_data=species, field_data=field, measure=kwargs['measure'])
    out = GData(tag=tag,
               label=kwargs['label'],
               comp_grid=ctx.obj['compgrid'],
               ctx=species.ctx)
    out.push(grid, agyro)
    data.add(out)
  #end

  verb_print(ctx, 'Finishing agyro')
#end
