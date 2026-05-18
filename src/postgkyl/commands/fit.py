import click
import numpy as np

from postgkyl.utils import verb_print
import postgkyl.tools as tools
from postgkyl.output.nodal_to_cell_centered_grid import nodal_to_cell_centered_grid


class FitTypeParam(click.ParamType):
  name = "fit_type"

  def convert(self, value, param, ctx):
    choices = list(tools.FIT_FUNCTIONS.keys())
    if value in choices:
      return value
    matches = [c for c in choices if c.startswith(value)]
    if len(matches) == 1:
      return matches[0]
    if len(matches) > 1:
      self.fail(f"'{value}' is ambiguous: matches {', '.join(sorted(matches))}", param, ctx)
    self.fail(f"'{value}' does not match any of: {', '.join(choices)}", param, ctx)

  def get_metavar(self, param, **_):
    return "{" + "|".join(tools.FIT_FUNCTIONS.keys()) + "}"


def _print_result(fit_type, params, std, R2):
  p = params
  s = std
  if fit_type == "linear":
    click.echo(
        f"Linear:      y = ({p[0]:.6e} ± {s[0]:.2e})*x"
        f" + ({p[1]:.6e} ± {s[1]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "quadratic":
    click.echo(
        f"Quadratic:   y = ({p[0]:.6e} ± {s[0]:.2e})*x²"
        f" + ({p[1]:.6e} ± {s[1]:.2e})*x"
        f" + ({p[2]:.6e} ± {s[2]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "plane":
    click.echo(
        f"Plane:       z = ({p[0]:.6e} ± {s[0]:.2e})*x"
        f" + ({p[1]:.6e} ± {s[1]:.2e})*y"
        f" + ({p[2]:.6e} ± {s[2]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "quadratic2d":
    click.echo(
        f"2D quadratic: z = ({p[0]:.6e} ± {s[0]:.2e})*x²"
        f" + ({p[1]:.6e} ± {s[1]:.2e})*y²"
        f" + ({p[2]:.6e} ± {s[2]:.2e})*x*y"
        f" + ({p[3]:.6e} ± {s[3]:.2e})*x"
        f" + ({p[4]:.6e} ± {s[4]:.2e})*y"
        f" + ({p[5]:.6e} ± {s[5]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "exp_plateau":
    click.echo(
        f"Exp plateau: y = ({p[0]:.6e} ± {s[0]:.2e})*exp(({p[1]:.6e} ± {s[1]:.2e})*x)"
        f" + ({p[2]:.6e} ± {s[2]:.2e})"
        f"    R² = {R2:.6f}"
    )


@click.command()
@click.argument("fit_type", type=FitTypeParam())
@click.option("--use", "-u", default=None, help="Specify a 'tag' to apply to. [default: all]")
@click.option("--guess", "-g", default=None, help="Comma-separated initial parameter guess.")
@click.option("--component", "-c", type=click.INT, default=0, show_default=True,
    help="Component index of the values array to fit.")
@click.pass_context
def fit(ctx, **kwargs):
  """Fit data with a model and print parameters + R².

  Model types (prefix-matched, same mechanism as pgkyl commands):
    linear      -- y = a*x + b
    quadratic   -- y = a*x² + b*x + c
    plane       -- z = a*x + b*y + c
    quadratic2d -- z = a*x² + b*y² + c*x*y + d*x + e*y + f
    exp_plateau -- y = A*exp(b*x) + C

  1D models require 1D data; 2D models require 2D data. Collapsed dimensions
  (e.g. after integrate) are automatically ignored. Does not modify the stack.
  """
  verb_print(ctx, "Starting fit")
  data = ctx.obj["data"]
  fit_type = FitTypeParam().convert(kwargs["fit_type"], None, None)
  ndim_fit = tools.FIT_NDIM[fit_type]

  for dat in data.iterator(kwargs["use"]):
    label = dat.get_label()
    tag = dat.get_tag()
    click.echo(click.style(f"{label} ({tag})" if label else tag, bold=True))

    grid = dat.get_grid()
    values = dat.get_values()

    spatial_shape = values.shape[:-1]
    if any(grid[d].shape[0] == spatial_shape[d] + 1 for d in range(len(grid))):
      cc_grid = nodal_to_cell_centered_grid(grid, spatial_shape)
    else:
      cc_grid = list(grid)

    # Drop dimensions collapsed to a single cell (e.g. after integrate / select)
    active = [d for d in range(len(cc_grid)) if cc_grid[d].shape[0] > 1]
    if len(active) < len(cc_grid):
      idx = tuple(slice(None) if d in active else 0
          for d in range(len(spatial_shape))) + (slice(None),)
      cc_grid = [cc_grid[d] for d in active]
      values = values[idx]

    n_spatial = len(cc_grid)

    if n_spatial != ndim_fit:
      ctx.fail(
          f"fit '{fit_type}' requires {ndim_fit} spatial dimension(s), "
          f"but data has {n_spatial}. Use 'select' or 'integrate' to reduce first."
      )

    ydata = values[..., kwargs["component"]].flatten()

    if ndim_fit == 1:
      xdata = cc_grid[0]
    else:
      X, Y = np.meshgrid(cc_grid[0], cc_grid[1], indexing="ij")
      xdata = np.array([X.flatten(), Y.flatten()])

    p0 = None
    if kwargs["guess"]:
      p0 = [float(v) for v in kwargs["guess"].split(",")]

    params, cov, R2 = tools.fit(xdata, ydata, fit_type, p0=p0)
    std = np.sqrt(np.diag(cov))
    _print_result(fit_type, params, std, R2)
