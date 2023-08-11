import click

# ---- Postgkyl imports ------------------------------------------------
from postgkyl.commands.util import verb_print
from postgkyl.data import GData
from postgkyl.tools import laguerre_compose
# ----------------------------------------------------------------------

@click.command(help='Compose PKPM Laguerre coefficients together.')
@click.option('--distribution', '-f', type=click.STRING, prompt=True,
              help='Specify the PKPM distribution function.')
@click.option('--moments', '-m', type=click.STRING, prompt=True,
              help='Specify the PKPM moments.')
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def laguerrecompose(ctx, **kwargs):
  verb_print(ctx, 'Starting laguerrecompose')
  data = ctx.obj['data']

  for f, mom in zip(data.iterator(kwargs['distribution']),
                    data.iterator(kwargs['moments'])):
    if kwargs['tag']:
      out = GData(tag=kwargs['tag'],
                  label=kwargs['label'],
                  comp_grid=ctx.obj['compgrid'],
                  ctx=f.ctx)
      laguerre_compose(f, mom, out)
    else:
      laguerre_compose(f, mom, f)
    #end
  #end
  verb_print(ctx, 'Finishing laguerrecompose')
#end
