import click

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools as diag


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option(
    "-v",
    "--variable_name",
    help="Variable to plot",
    prompt=True,
    type=click.Choice(
        [
            "density",
            "xvel",
            "yvel",
            "zvel",
            "vel",
            "pressureTensor",
            "pxx",
            "pxy",
            "pxz",
            "pyy",
            "pyz",
            "pzz",
            "pressure",
            "temp",
            "ke",
            "sound",
            "mach",
        ]
    ),
)
@click.option(
    "-g",
    "--gas_gamma",
    help="Gas adiabatic constant",
    type=click.FLOAT,
    default=5.0 / 3.0,
)
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def tenmoment(ctx, **kwargs):
  """Extract ten-moment primitive variables from ten-moment conserved
  variables.
  """
  verb_print(ctx, "Starting tenmoment")
  data = ctx.obj["data"]

  v = kwargs["variable_name"]
  for dat in data.iterator(kwargs["use"]):
    verb_print(ctx, "tenmoment: Extracting {:s} from data set".format(v))
    out = dat
    if kwargs["tag"]:
      out = GData(
          tag=kwargs["tag"],
          label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"],
          ctx=dat.ctx,
      )
      data.add(out)
    # end
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
    elif v == "pressureTensor":
      diag.get_pij(dat, out_mom=out)
    elif v == "pxx":
      diag.get_pxx(dat, out_mom=out)
    elif v == "pxy":
      diag.get_pxy(dat, out_mom=out)
    elif v == "pxz":
      diag.get_pxz(dat, out_mom=out)
    elif v == "pyy":
      diag.get_pyy(dat, out_mom=out)
    elif v == "pyz":
      diag.get_pyz(dat, out_mom=out)
    elif v == "pzz":
      diag.get_pzz(dat, out_mom=out)
    elif v == "pressure":
      diag.get_p(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "ke":
      diag.get_ke(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "temp":
      diag.get_temp(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "sound":
      diag.get_sound(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    elif v == "mach":
      diag.get_mach(dat, gas_gamma=kwargs["gas_gamma"], num_moms=10, out_mom=out)
    # end
  # end
  verb_print(ctx, "Finishing tenmoment")
