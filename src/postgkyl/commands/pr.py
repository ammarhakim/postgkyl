import click
import numpy as np

from postgkyl.utils import verb_print

np.set_printoptions(precision=16)

@click.command(help="Print the data")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--grid", "-g", is_flag=True, help="Print grid instead of values.")
@click.pass_context
def pr(ctx, **kwargs):
  verb_print(ctx, "Starting pr")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    if kwargs["grid"]:
      grid = dat.get_grid()
      for g in grid:
        click.echo(g)
      # end
    else:
      click.echo(dat.get_values().squeeze())
    # end
  # end

  verb_print(ctx, "Finishing pr")
