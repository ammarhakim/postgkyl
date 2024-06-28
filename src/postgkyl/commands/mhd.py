import click

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools as diag


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option(
    "-m", "--mu0", help="Permeability of free space", type=click.FLOAT, default=1.0
)
@click.option(
    "-g",
    "--gas_gamma",
    help="Gas adiabatic constant",
    type=click.FLOAT,
    default=5.0 / 3.0,
)
@click.option(
    "-v",
    "--variable_name",
    help="Variable to extract",
    prompt=True,
    type=click.Choice(
        [
            "density",
            "xvel",
            "yvel",
            "zvel",
            "vel",
            "Bx",
            "By",
            "Bz",
            "Bi",
            "magpressure",
            "pressure",
            "temp",
            "sound",
            "mach",
        ]
    ),
)
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def mhd(ctx, **kwargs):
  """Compute ideal MHD primitive and some derived variables from MHD
  conserved variables.

  """
  verb_print(ctx, "Starting mhd")
  data = ctx.obj["data"]

  v = kwargs["variable_name"]
  for dat in data.iterator(kwargs["use"]):
    verb_print(ctx, "mhd: Extracting {:s} from data set".format(v))
    overwrite = True
    if kwargs["tag"]:
      overwrite = False
      out = GData(
          tag=kwargs["tag"],
          label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"],
          ctx=dat.ctx,
      )
    # end
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
    elif v == "Bx":
      grid, values = diag.get_mhd_Bx(dat, overwrite=overwrite)
    elif v == "By":
      grid, values = diag.get_mhd_By(dat, overwrite=overwrite)
    elif v == "Bz":
      grid, values = diag.get_mhd_Bz(dat, overwrite=overwrite)
    elif v == "Bi":
      grid, values = diag.get_mhd_Bi(dat, overwrite=overwrite)
    elif v == "magpressure":
      grid, values = diag.get_mhd_mag_p(
          dat, gasGamma=kwargs["gas_gamma"], mu0=kwargs["mu0"], overwrite=overwrite
      )
    elif v == "pressure":
      grid, values = diag.get_mhd_p(
          dat, gasGamma=kwargs["gas_gamma"], mu0=kwargs["mu0"], overwrite=overwrite
      )
    elif v == "temp":
      grid, values = diag.get_mhd_temp(
          dat, gasGamma=kwargs["gas_gamma"], mu0=kwargs["mu0"], overwrite=overwrite
      )
    elif v == "sound":
      grid, values = diag.get_mhd_sound(
          dat, gasGamma=kwargs["gas_gamma"], mu0=kwargs["mu0"], overwrite=overwrite
      )
    elif v == "mach":
      grid, values = diag.get_mhd_mach(
          dat, gasGamma=kwargs["gas_gamma"], mu0=kwargs["mu0"], overwrite=overwrite
      )
    # end
    if kwargs["tag"]:
      out.push(grid, values)
      data.add(out)
    # end
  # end
  verb_print(ctx, "Finishing mhd")
