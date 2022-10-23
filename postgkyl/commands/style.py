import click
import shutil
import os.path

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Probe and control the plotting style.')
@click.option('--print', '-p', is_flag=True,
              help="Prints the current rcParams.")
@click.pass_context
def style(ctx, **kwargs):
  vlog(ctx, 'Starting style')
  pushChain(ctx, 'style', **kwargs)


  if kwargs['print']:
    for key in ctx.obj['rcParams']:
      print('{:s} : {}'.format(key, ctx.obj['rcParams'][key]))
    #end
  
  vlog(ctx, 'Finishing style')
#end
