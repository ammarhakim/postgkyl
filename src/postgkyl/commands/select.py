import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.data.select


@click.command()
@click.option("--z0", default=None, help="Indices for 0th coord (either int, float, or slice).")
@click.option("--z1", default=None, help="Indices for 1st coord (either int, float, or slice).")
@click.option("--z2", default=None, help="Indices for 2nd coord (either int, float, or slice).")
@click.option("--z3", default=None, help="Indices for 3rd coord (either int, float, or slice).")
@click.option("--z4", default=None, help="Indices for 4th coord (either int, float, or slice).")
@click.option("--z5", default=None, help="Indices for 5th coord (either int, float, or slice).")
@click.option("--comp", "-c", default=None,
    help="Indices for components (either int, slice, or coma-separated).")
@click.option("--use", "-u", help="Specify a 'tag' to apply to.")
@click.option("--tag", "-t", help="Optional tag for the resulting array.")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def select(ctx, **kwargs):
  """Subselect data from the active dataset(s).

  This command allows, for example, to choose a specific component of a multi-component
  dataset, select a index or coordinate range. Index ranges can also be specified using
  python slice notation (start:end:stride).
  """
  verb_print(ctx, "Starting select")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      grid, values = postgkyl.data.select(dat,
          z0=kwargs["z0"], z1=kwargs["z1"], z2=kwargs["z2"], z3=kwargs["z3"],
          z4=kwargs["z4"], z5=kwargs["z5"], comp=kwargs["comp"])
      out.push(grid, values)
      data.add(out)
    else:
      postgkyl.data.select(dat, overwrite=True,
          z0=kwargs["z0"], z1=kwargs["z1"], z2=kwargs["z2"], z3=kwargs["z3"],
          z4=kwargs["z4"], z5=kwargs["z5"], comp=kwargs["comp"])
    # end
  # end
  verb_print(ctx, "Finishing select")
