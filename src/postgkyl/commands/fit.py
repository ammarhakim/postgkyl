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


def _auto_guess(fit_type, xdata, ydata):
  """Return data-driven initial parameter guesses for known fit types."""
  y = np.asarray(ydata, dtype=float)
  finite = np.isfinite(y)
  if not np.any(finite):
    return None
  y_fin = y[finite]
  y_min, y_max = y_fin.min(), y_fin.max()
  y_mean = y_fin.mean()
  y_range = y_max - y_min

  if fit_type == "linear":
    x = np.asarray(xdata)
    dx = x.max() - x.min()
    a = y_range / dx if dx != 0 else 1.0
    b = y_mean - a * x.mean()
    return [a, b]

  if fit_type == "quadratic":
    x = np.asarray(xdata)
    try:
      return list(np.polyfit(x, y, 2))
    except Exception:
      return [0.0, 1.0, y_mean]

  if fit_type == "plane":
    x, yc = xdata[0], xdata[1]
    A = np.column_stack([x, yc, np.ones_like(x)])
    result, *_ = np.linalg.lstsq(A, y, rcond=None)
    return list(result)

  if fit_type == "quadratic2d":
    x, yc = xdata[0], xdata[1]
    A = np.column_stack([x**2, yc**2, x * yc, x, yc, np.ones_like(x)])
    result, *_ = np.linalg.lstsq(A, y, rcond=None)
    return list(result)

  if fit_type == "exp_plateau":
    x = np.asarray(xdata)
    n_tail = max(1, len(x) // 10)
    C = float(y[np.argsort(x)[-n_tail:]].mean())
    A = float(y_max - C) or 1.0
    x_span = x.max() - x.min()
    b = -1.0 / x_span if x_span > 0 else -1.0
    return [A, b, C]

  if fit_type == "gaussian":
    x = np.asarray(xdata)
    A = float(y_max)
    mu = float(x[np.argmax(y)])
    above = x[y >= A / 2] if A != 0 else x
    if len(above) >= 2:
      sigma = float((above[-1] - above[0]) / (2 * np.sqrt(2 * np.log(2))))
    else:
      sigma = float((x.max() - x.min()) / 4)
    return [A, mu, max(abs(sigma), 1e-10)]

  if fit_type == "power":
    b_off = float(y_min)
    a = float(y_max - b_off) or 1.0
    return [a, 1.0, b_off]

  if fit_type == "sinusoid":
    x = np.asarray(xdata)
    A = float(y_range / 2) or 1.0
    C = float((y_max + y_min) / 2)
    sort_idx = np.argsort(x)
    x_s, y_s = x[sort_idx], y[sort_idx]
    if len(x_s) > 1:
      dx = np.mean(np.diff(x_s))
      freqs = np.fft.rfftfreq(len(y_s), d=dx)
      fft_amp = np.abs(np.fft.rfft(y_s - C))
      i_peak = np.argmax(fft_amp[1:]) + 1 if len(fft_amp) > 1 else 1
      omega = float(2 * np.pi * freqs[i_peak])
    else:
      omega = 1.0
    return [A, omega, 0.0, C]

  if fit_type == "tanh_transition":
    x = np.asarray(xdata)
    A = float(y_range / 2) or 1.0
    C = float((y_max + y_min) / 2)
    x0 = float(x[np.argmax(np.abs(np.gradient(y)))])
    w = float((x.max() - x.min()) / 4) or 1.0
    return [A, x0, w, C]

  return None


@click.command()
@click.argument("fit_type", type=FitTypeParam())
@click.option("--use", "-u", default=None, help="Specify a 'tag' to apply to. [default: all]")
@click.option("--guess", "-g", default=None, help="Comma-separated initial parameter guess.")
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

    if ndim_fit == 1:
      xdata = cc_grid[0]
    else:
      X, Y = np.meshgrid(cc_grid[0], cc_grid[1], indexing="ij")
      xdata = np.array([X.flatten(), Y.flatten()])

    user_p0 = None
    if kwargs["guess"]:
      user_p0 = [float(v) for v in kwargs["guess"].split(",")]

    n_components = values.shape[-1]
    active_spatial_shape = tuple(cg.shape[0] for cg in cc_grid)
    fit_values_list = []
    for comp in range(n_components):
      if n_components > 1:
        click.echo(f"  Component {comp}:")
      ydata = values[..., comp].flatten()
      p0 = user_p0 if user_p0 is not None else _auto_guess(fit_type, xdata, ydata)
      params, cov, R2 = tools.fit(xdata, ydata, fit_type, p0=p0)
      std = np.sqrt(np.diag(cov))
      _print_result(fit_type, params, std, R2)
      y_fit = tools.fit_evaluate(xdata, fit_type, params)
      fit_values_list.append(y_fit.reshape(active_spatial_shape + (1,)))

    fit_values = np.concatenate(fit_values_list, axis=-1)
    fit_grid = [grid[d] for d in active]
    out = GData(tag=dat.get_tag())
    out.push(fit_grid, fit_values)
    data.add(out)
