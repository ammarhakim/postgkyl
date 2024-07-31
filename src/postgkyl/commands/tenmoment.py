import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools.prim_vars as pv


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("-v", "--variable_name", prompt=True,
    type=click.Choice(["density", "xvel", "yvel", "zvel", "vel", "pressureTensor",
        "pxx", "pxy", "pxz", "pyy", "pyz", "pzz", "pressure", "temp", "ke", "sound", "mach"]),
    help="Variable to work with.")
@click.option("-g", "--gas_gamma",type=click.FLOAT, show_default=True, default=5.0/3,
    help="Gas adiabatic constant.")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def tenmoment(ctx, **kwargs):
  """Extract ten-moment primitive variables from ten-moment conserved variables.
  """
  verb_print(ctx, "Starting tenmoment")
  data = ctx.obj["data"]

  v = kwargs["variable_name"]
  for dat in data.iterator(kwargs["use"]):
    verb_print(ctx, f"tenmoment: Extracting {v:s} from data set")
    out = dat
    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      data.add(out)
    # end
    if v == "density":
      pv.get_density(dat, out_mom=out)
    elif v == "xvel":
      pv.get_vx(dat, out_mom=out)
    elif v == "yvel":
      pv.get_vy(dat, out_mom=out)
    elif v == "zvel":
      pv.get_vz(dat, out_mom=out)
    elif v == "vel":
      pv.get_vi(dat, out_mom=out)
    elif v == "pressureTensor":
      pv.get_pij(dat, out_mom=out)
    elif v == "pxx":
      pv.get_pxx(dat, out_mom=out)
    elif v == "pxy":
      pv.get_pxy(dat, out_mom=out)
    elif v == "pxz":
      pv.get_pxz(dat, out_mom=out)
    elif v == "pyy":
      pv.get_pyy(dat, out_mom=out)
    elif v == "pyz":
      pv.get_pyz(dat, out_mom=out)
    elif v == "pzz":
      pv.get_pzz(dat, out_mom=out)
    elif v == "pressure":
      pv.get_p(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "ke":
      pv.get_ke(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "temp":
      pv.get_temp(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "sound":
      pv.get_sound(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "mach":
      pv.get_mach(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    # end
  # end
  verb_print(ctx, "Finishing tenmoment")
