import click
import numpy as np

from postgkyl.utils import verb_print



# ---- Math ----
@click.command(help="Multiply data by a factor")
@click.argument("factor", nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, **kwargs):
  verb_print(ctx, f"Multiplying by {kwargs['factor']:f}")
  for s in ctx.obj["sets"]:
    values = ctx.obj["dataSets"][s].get_values()
    values = values * kwargs["factor"]
    ctx.obj["dataSets"][s].push(values)
  # end


@click.command(help="Calculate power of data")
@click.argument("power", nargs=1, type=click.FLOAT)
@click.pass_context
def pow(ctx, **kwargs):
  verb_print(ctx, f"Calculating the power of {kwargs['power']:f}")
  for s in ctx.obj["sets"]:
    values = ctx.obj["dataSets"][s].get_values()
    values = values ** kwargs["power"]
    ctx.obj["dataSets"][s].push(values)
  # end


@click.command(help="Calculate natural log of data")
@click.pass_context
def log(ctx):
  verb_print(ctx, "Calculating the natural log")
  for s in ctx.obj["sets"]:
    values = ctx.obj["dataSets"][s].get_values()
    values = np.log(values)
    ctx.obj["dataSets"][s].push(values)
  # end


@click.command(help="Calculate absolute values of data")
@click.pass_context
def abs(ctx):
  verb_print(ctx, "Calculating the absolute value")
  for s in ctx.obj["sets"]:
    values = ctx.obj["dataSets"][s].get_values()
    values = np.abs(values)
    ctx.obj["dataSets"][s].push(values)
  # end


@click.command(help="Normalize data")
@click.option("--shift/--no-shift", default=False, show_default=True,
    help="Shift minimal value to zero.")
@click.option("--usefirst", is_flag=True, default=False, help="Normalize to first value in field.")
@click.pass_context
def norm(ctx, **kwargs):
  verb_print(ctx, "Normalizing data")
  for s in ctx.obj["sets"]:
    values = ctx.obj["dataSets"][s].get_values()
    num_comps = ctx.obj["dataSets"][s].get_num_comps()
    values_out = values.copy()
    for comp in range(num_comps):
      if kwargs["shift"]:
        values_out[..., comp] -= values_out[..., comp].min()
      if kwargs["usefirst"]:
        values_out[..., comp] /= values_out[..., comp].item(0)
      else:
        values_out[..., comp] /= np.abs(values_out[..., comp]).max()
      # end
    # end
    ctx.obj["dataSets"][s].push(values_out)
  # end
