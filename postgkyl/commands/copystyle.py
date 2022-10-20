import click
import shutil
import os.path

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print info of active datasets.')
@click.argument('filename')
@click.pass_context
def copystyle(ctx, **kwargs):
  vlog(ctx, 'Starting copystyle')
  pushChain(ctx, 'copystyle', **kwargs)
  
  out_fn = kwargs["filename"]
  if len(out_fn.split('.')) == 1:
    out_fn = '{:s}.mplstyle'.format(out_fn)
  elif out_fn.split('.')[1] != 'mplstyle':
    out_fn = '{:s}.mplstyle'.format(out_fn)
  #end
  pstyle_src = '{:s}/../output/postgkyl.mplstyle'.format(os.path.dirname(os.path.realpath(__file__)))

  shutil.copyfile(pstyle_src, out_fn)
  
  vlog(ctx, 'Finishing copystyle')
#end
