import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools.prim_vars as pv


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--mu0", "-m", type=click.FLOAT, default=1.0, show_default=True,
    help="Permeability of free space.")
@click.option("--gas_gamma", "-g", type=click.FLOAT, default=5.0/3, show_default=True,
    help="Gas adiabatic constant.")
@click.option("--variable_name", "-v", prompt=True,
    type=click.Choice(["density", "xvel", "yvel", "zvel", "vel", "Bx", "By", "Bz", "Bi",
        "magpressure", "pressure", "temp", "sound", "mach"]),
    help="Variable to extract")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def mhd(ctx, **kwargs):
  """Compute ideal MHD primitive and some derived variables from MHD conserved variables.
  """
  verb_print(ctx, "Starting mhd")
  data = ctx.obj["data"]

  v = kwargs["variable_name"]
  for dat in data.iterator(kwargs["use"]):
    verb_print(ctx, f"mhd: Extracting {v:s} from data set")
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
    elif v == "Bx":
      pv.get_mhd_Bx(dat, out_mom=out)
    elif v == "By":
      pv.get_mhd_By(dat, out_mom=out)
    elif v == "Bz":
      pv.get_mhd_Bz(dat, out_mom=out)
    elif v == "Bi":
      pv.get_mhd_Bi(dat, out_mom=out)
    elif v == "magpressure":
      pv.get_mhd_mag_p(dat, mu_0=kwargs["mu0"], out_mom=out)
    elif v == "pressure":
      pv.get_mhd_p(dat, gas_gamma=kwargs["gas_gamma"], mu_0=kwargs["mu0"], out_mom=out)
    elif v == "temp":
      pv.get_mhd_temp(dat, gas_gamma=kwargs["gas_gamma"], mu_0=kwargs["mu0"], out_mom=out)
    elif v == "sound":
      pv.get_mhd_sound(dat, gas_gamma=kwargs["gas_gamma"], mu_0=kwargs["mu0"], out_mom=out)
    elif v == "mach":
      pv.get_mhd_mach(dat, gas_gamma=kwargs["gas_gamma"], mu_0=kwargs["mu0"], out_mom=out)
    # end
  # end
  verb_print(ctx, "Finishing mhd")
