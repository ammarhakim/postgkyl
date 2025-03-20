import click
import numpy as np

import postgkyl.tools as diag
from postgkyl.data import GData
from postgkyl.commands.util import verb_print

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-m', '--mu0', help="Permeability of free space",
              type=click.FLOAT, default=1.0)
@click.option('-g', '--gas_gamma', help="Gas adiabatic constant",
              type=click.FLOAT, default=5.0/3.0)
@click.option('-v', '--variable_name', help="Variable to extract", prompt=True,
              type=click.Choice(["density", "xvel", "yvel",
                                 "zvel", "vel", "Bx", "By", "Bz", "Bi",
                                 "magpressure", "pressure", "temp", "sound", "mach"]))
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def mhd(ctx, **kwargs):
  """Compute ideal MHD primitive and some derived variables from MHD
    conserved variables.

  """
  verb_print(ctx, 'Starting mhd')
  data = ctx.obj['data']

  v = kwargs['variable_name']
  for dat in data.iterator(kwargs['use']):
    verb_print(ctx, 'mhd: Extracting {:s} from data set'.format(v))
    out = dat
    if kwargs['tag']:
      overwrite = False
      out = GData(tag=kwargs['tag'],
                  label=kwargs['label'],
                  comp_grid=ctx.obj['compgrid'],
                  ctx=dat.ctx)
      data.add(out)
    #end
    if v == "density":
      diag.get_density(dat, out_mom=out)
    elif v == "xvel":
      diag.get_vx(dat, out_mom=out)
    elif v == "yvel":
      diag.get_vy(dat, out_mom=out)
    elif v == "zvel":
      diag.get_vz(dat, out_mom=out)
    elif v == "vel":
      diag.get_vi(dat, out_mom=out)
    elif v == "Bx":
      diag.get_mhd_Bx(dat, out_mom=out)
    elif v == "By":
      diag.get_mhd_By(dat, out_mom=out)
    elif v == "Bz":
      diag.get_mhd_Bz(dat, out_mom=out)
    elif v == "Bi":
      diag.get_mhd_Bi(dat, out_mom=out)
    elif v == "magpressure":
      diag.get_mhd_mag_p(dat, mu_0=kwargs['mu0'], out_mom=out)
    elif v == "pressure":
      diag.get_mhd_p(dat, gas_gamma=kwargs['gas_gamma'], mu_0=kwargs['mu0'], out_mom=out)
    elif v == "temp":
      diag.get_mhd_temp(dat, gas_gamma=kwargs['gas_gamma'], mu_0=kwargs['mu0'], out_mom=out)
    elif v == "sound":
      diag.get_mhd_sound(dat, gas_gamma=kwargs['gas_gamma'], mu_0=kwargs['mu0'], out_mom=out)
    elif v == "mach":
      diag.get_mhd_mach(dat, gas_gamma=kwargs['gas_gamma'], mu_0=kwargs['mu0'], out_mom=out)
    #end
  verb_print(ctx, 'Finishing mhd')
#end
