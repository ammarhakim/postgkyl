import click
import numpy as np

import postgkyl.diagnostics as diag
from postgkyl.data import Data
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to extract", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressure", "ke", "mach"]))
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def euler(ctx, **kwargs):
  """Compute Euler (five-moment) primitive and some derived variables
    from fluid conserved variables.
  """
  vlog(ctx, 'Starting euler')
  pushChain(ctx, 'euler', **kwargs)
  data = ctx.obj['data']
  
  v = kwargs['variable_name']
  for dat in data.iterator(kwargs['use']):
    vlog(ctx, 'euler: Extracting {:s} from data set'.format(v))
    if kwargs['tag']:
      out = Data(tag=kwargs['tag'],
                label=kwargs['label'],
                 comp_grid=ctx.obj['compgrid'],
                 meta=dat.meta)
      if v == "density":
        grid, values = diag.getDensity(dat)
      elif v == "xvel":
        grid, values = diag.getVx(dat)
      elif v == "yvel":
        grid, values = diag.getVy(dat)
      elif v == "zvel":
        grid, values = diag.getVz(dat)
      elif v == "vel":
        grid, values = diag.getVi(dat)
      elif v == "pressure":
        grid, values = diag.getP(dat, gasGamma=kwargs['gas_gamma'], numMom=5)
      elif v == "ke":
        grid, values = diag.getKE(dat, gasGamma=kwargs['gas_gamma'], numMom=5)
      elif v == "mach":
        grid, values = diag.getMach(dat, gasGamma=kwargs['gas_gamma'], numMom=5)
      #end
      out.push(grid, values)
      data.add(out)
    else:
      if v == "density":
        diag.getDensity(dat, overwrite=True)
      elif v == "xvel":
        diag.getVx(dat, overwrite=True)
      elif v == "yvel":
        diag.getVy(dat, overwrite=True)
      elif v == "zvel":
        diag.getVz(dat, overwrite=True)
      elif v == "vel":
        diag.getVi(dat, overwrite=True)
      elif v == "pressure":
        diag.getP(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
      elif v == "ke":
        diag.getKE(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
      elif v == "mach":
        diag.getMach(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=True)
      #end
    #end
  #end
  vlog(ctx, 'Finishing euler')
#end
    
