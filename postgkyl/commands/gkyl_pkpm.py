import click

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.commands.util import verb_print
from postgkyl.data import GData, GInterpModal
from postgkyl.tools import laguerre_compose
from postgkyl.tools import transform_frame
# ----------------------------------------------------------------------

@click.command(help='Shortcut to load Gkeyll PKPM data, interpolate, and transform')
@click.option('--name', '-n', type=click.STRING, prompt=True,
              help="Set the root name for files")
@click.option('--species', '-s', type=click.STRING, prompt=True,
              help="Set species name")
@click.option('--idx', '-i', type=click.STRING, prompt=True,
              help="Set the file number")
@click.option('--polyorder', '-p', type=click.INT, prompt=True,
              help="Set the polynomial order")
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def gkylpkpm(ctx, **kwargs):
  verb_print(ctx, 'Starting Gkyl PKPM')
  data = ctx.obj['data']

  gf = GData('{:s}-{:s}_{:s}.gkyl'.format(
    kwargs['name'], kwargs['species'], kwargs['idx']))
  gmoms = GData('{:s}-{:s}_pkpm_moms_{:s}.gkyl'.format(
    kwargs['name'], kwargs['species'], kwargs['idx']))
  gu = GData('{:s}-fluid_{:s}_u_{:s}.gkyl'.format(
    kwargs['name'], kwargs['species'], kwargs['idx']))

  num_dims = gf.getNumDims()
  c_dim = num_dims - 1

  dg = GInterpModal(gf, kwargs['polyorder'], 'pkpmhyb')
  dg.interpolate((0,1), overwrite=True)

  dg = GInterpModal(gmoms, kwargs['polyorder'], 'ms')
  dg.interpolate((0,1,2,3,4,5,6), overwrite=True)

  dg = GInterpModal(gu, kwargs['polyorder'], 'ms')
  dg.interpolate((0,1,2), overwrite=True)

  laguerre_compose(gf, gmoms, gf)
  transform_frame(gf, gu, c_dim, gf)

  gf.setTag(kwargs['tag'])
  gf.setLabel(kwargs['label'])
  data.add(gf)

  verb_print(ctx, 'Finishing Gkyl PKPM')
#end