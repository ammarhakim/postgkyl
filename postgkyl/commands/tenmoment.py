import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-v', '--variable_name', help="Variable to plot", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressureTensor",
                                 "pxx", "pxy", "pxz", "pyy", "pyz", "pzz",
                                 "pressure", "ke", "sound", "mach"]))
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def tenmoment(ctx, **kwargs):
  """Extract ten-moment primitive variables from ten-moment conserved
  variables.
  """
  vlog(ctx, 'Starting tenmoment')
  pushChain(ctx, 'tenmoment', **kwargs)
  data = ctx.obj['data']
  
  v = kwargs['variable_name']
  for dat in data.iterator(kwargs['use']):
    vlog(ctx, 'tenmoment: Extracting {:s} from data set'.format(v))
    overwrite = True
    if kwargs['tag']:
      overwrite = False
      out = Data(tag=kwargs['tag'],
                label=kwargs['label'],
                 comp_grid=ctx.obj['compgrid'],
                 meta=dat.meta)
    #end
    if v == "density":
      grid, values = diag.get_density(dat, overwrite=overwrite)
    elif v == "xvel":
      grid, values = diag.get_vx(dat, overwrite=overwrite)
    elif v == "yvel":
      grid, values = diag.get_vy(dat, overwrite=overwrite)
    elif v == "zvel":
      grid, values = diag.get_vz(dat, overwrite=overwrite)
    elif v == "vel":
      grid, values = diag.get_vi(dat, overwrite=overwrite)
    elif v == "pressureTensor":
      grid, values = diag.get_pij(dat, overwrite=overwrite)
    elif v == "pxx":
      grid, values = diag.get_pxx(dat, overwrite=overwrite)
    elif v == "pxy":
      grid, values = diag.get_pxy(dat, overwrite=overwrite)
    elif v == "pxz":
      grid, values = diag.get_pxz(dat, overwrite=overwrite)
    elif v == "pyy":
      grid, values = diag.get_pyy(dat, overwrite=overwrite)
    elif v == "pyz":
      grid, values = diag.get_pyz(dat, overwrite=overwrite)
    elif v == "pzz":
      grid, values = diag.get_pzz(dat, overwrite=overwrite)
    elif v == "pressure":
      grid, values = diag.get_p(dat, numMom=10, overwrite=overwrite)
    elif v == "ke":
      grid, values = diag.get_ke(dat, numMom=10, overwrite=overwrite)
    elif v == "sound":
      grid, values = diag.get_sound(dat, numMom=10, overwrite=overwrite)
    elif v == "mach":
      grid, values = diag.get_mach(dat, numMom=10, overwrite=overwrite)
    #end
    if kwargs['tag']:
      out.push(grid, values)
      data.add(out)
    #end
  #end
  vlog(ctx, 'Finishing tenmoment')
#end
