import click
import numpy as np

from postgkyl.data.gdata import GData
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
    # not a known type — accept if it looks like an RPN expression
    toks = set(value.split())
    if toks & (tools.RPN_OPERATORS | set(tools.RPN_FUNCTIONS)):
      return value
    self.fail(
        f"'{value}' does not match any known fit type ({', '.join(choices)}) "
        f"and is not a valid RPN expression (must contain at least one operator or function).",
        param, ctx,
    )

  def get_metavar(self, param, **_):
    return "{" + "|".join(tools.FIT_FUNCTIONS.keys()) + "|<rpn-expr>}"


def _print_result(fit_type, params, std, R2, param_names=None):
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
  elif fit_type == "gaussian":
    click.echo(
        f"Gaussian:    y = ({p[0]:.6e} ± {s[0]:.2e})"
        f"*exp(-0.5*((x - ({p[1]:.6e} ± {s[1]:.2e}))/({p[2]:.6e} ± {s[2]:.2e}))²)"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "power":
    click.echo(
        f"Power law:   y = ({p[0]:.6e} ± {s[0]:.2e})*x^({p[1]:.6e} ± {s[1]:.2e})"
        f" + ({p[2]:.6e} ± {s[2]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "sinusoid":
    click.echo(
        f"Sinusoid:    y = ({p[0]:.6e} ± {s[0]:.2e})"
        f"*sin(({p[1]:.6e} ± {s[1]:.2e})*x + ({p[2]:.6e} ± {s[2]:.2e}))"
        f" + ({p[3]:.6e} ± {s[3]:.2e})"
        f"    R² = {R2:.6f}"
    )
  elif fit_type == "tanh_transition":
    click.echo(
        f"Tanh:        y = ({p[0]:.6e} ± {s[0]:.2e})"
        f"*tanh((x - ({p[1]:.6e} ± {s[1]:.2e}))/({p[2]:.6e} ± {s[2]:.2e}))"
        f" + ({p[3]:.6e} ± {s[3]:.2e})"
        f"    R² = {R2:.6f}"
    )
  else:
    names = param_names or tools.rpn_param_names(fit_type)
    parts = "  ".join(f"{n} = {p[i]:.6e} ± {s[i]:.2e}" for i, n in enumerate(names))
    click.echo(f"Custom ({fit_type}):  {parts}    R² = {R2:.6f}")


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
    linear          -- y = a*x + b
    quadratic       -- y = a*x² + b*x + c
    plane           -- z = a*x + b*y + c  [2D]
    quadratic2d     -- z = a*x² + b*y² + c*x*y + d*x + e*y + f  [2D]
    exp_plateau     -- y = A*exp(b*x) + C
    gaussian        -- y = A*exp(-0.5*((x-mu)/sigma)²)
    power           -- y = a*x^n + b
    sinusoid        -- y = A*sin(omega*x + phi) + C
    tanh_transition -- y = A*tanh((x-x0)/w) + C

  A custom model can also be given as a Reverse Polish Notation expression.
  x (and y for 2D) are the spatial variables; all other identifiers are free
  parameters.  Supported operators: + - * / ** ^.  Supported functions:
  exp log ln log10 sin cos tan sqrt abs tanh.

  Example:  fit 'a x * b +'   fits y = a*x + b

  1D models require 1D data; 2D models require 2D data. Collapsed dimensions
  (e.g. after integrate) are automatically ignored. Adds the fitted curve as a
  new dataset on the stack (same tag, same nodal grid, values at cell centers).
  """
  verb_print(ctx, "Starting fit")
  data = ctx.obj["data"]
  fit_type = FitTypeParam().convert(kwargs["fit_type"], None, None)
  ndim_fit = tools.FIT_NDIM.get(fit_type, tools.rpn_ndim(fit_type))

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

    y_fit = tools.fit_evaluate(xdata, fit_type, params)
    active_spatial_shape = tuple(cg.shape[0] for cg in cc_grid)
    fit_values = y_fit.reshape(active_spatial_shape + (1,))
    fit_grid = [grid[d] for d in active]
    out = GData(tag=dat.get_tag())
    out.push(fit_grid, fit_values)
    data.add(out)
