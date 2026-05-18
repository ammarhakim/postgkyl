import click
import numpy as np

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools as tools


@click.command()
@click.argument("fit_type", type=click.Choice(["linear", "quadratic"]))
@click.option("--use", "-u", default=None, help="Specify a 'tag' to apply to. [default: all]")
@click.option("--guess", "-g", help="Comma-separated initial parameter guess.")
@click.option("--component", "-c", type=click.INT, default=0, show_default=True,
    help="Component index of the values array to fit.")
@click.option("--tag", "-t", help="Tag for a new dataset containing the fit curve.")
@click.option("--label", "-l", help="Custom label for the resulting dataset.")
@click.pass_context
def fit(ctx, **kwargs):
  """Fit data with a polynomial model.

  FIT_TYPE is one of: linear (y = a*x + b) or quadratic (y = a*x^2 + b*x + c).

  Prints the fit parameters and R² to stdout. With --tag, also creates a new
  dataset containing the fitted curve evaluated on the same grid.
  """
  verb_print(ctx, "Starting fit")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    grid = dat.get_grid()
    values = dat.get_values()

    x = grid[0]
    y = values[..., kwargs["component"]].squeeze()

    p0 = None
    if kwargs["guess"]:
      p0 = [float(v) for v in kwargs["guess"].split(",")]

    params, cov, R2 = tools.fit(x, y, kwargs["fit_type"], p0=p0)
    std = np.sqrt(np.diag(cov))

    fit_type = kwargs["fit_type"]
    if fit_type == "linear":
      click.echo(
          f"Linear fit:    y = ({params[0]:.6e} ± {std[0]:.2e})*x"
          f" + ({params[1]:.6e} ± {std[1]:.2e})"
          f"    R² = {R2:.6f}"
      )
    elif fit_type == "quadratic":
      click.echo(
          f"Quadratic fit: y = ({params[0]:.6e} ± {std[0]:.2e})*x²"
          f" + ({params[1]:.6e} ± {std[1]:.2e})*x"
          f" + ({params[2]:.6e} ± {std[2]:.2e})"
          f"    R² = {R2:.6f}"
      )

    if kwargs["tag"]:
      y_fit = tools.FIT_FUNCTIONS[fit_type](x, *params)
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      out.push([x], y_fit[..., np.newaxis])
      data.add(out)

  verb_print(ctx, "Finishing fit")
