import click
import numpy as np

import postgkyl.tools as diag
from postgkyl.data import GData
from postgkyl.commands.util import verb_print

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to extract", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "pressure", "ke", "temp", "sound", "mach"]))
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def euler(ctx, **kwargs):
  """Compute Euler (five-moment) primitive and some derived variables
    from fluid conserved variables.
  """
  verb_print(ctx, 'Starting euler')
  data = ctx.obj['data']

  v = kwargs['variable_name']
  for dat in data.iterator(kwargs['use']):
    verb_print(ctx, 'euler: Extracting {:s} from data set'.format(v))
    overwrite = True
    if kwargs['tag']:
      overwrite = False
      out = GData(tag=kwargs['tag'],
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
    elif v == "pressure":
      grid, values = diag.get_p(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=overwrite)
    elif v == "ke":
      grid, values = diag.get_ke(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=overwrite)
    elif v == "temp":
      grid, values = diag.get_temp(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=overwrite)
    elif v == "sound":
      grid, values = diag.get_sound(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=overwrite)
    elif v == "mach":
      grid, values = diag.get_mach(dat, gasGamma=kwargs['gas_gamma'], numMom=5, overwrite=overwrite)
    #end
    if kwargs['tag']:
      out.push(grid, values)
      data.add(out)
    #end
  #end
  verb_print(ctx, 'Finishing euler')
#end

