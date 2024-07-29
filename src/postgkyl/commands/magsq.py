import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools


@click.command()
@click.option("--use", "-u", default=None, help="Specify the tag to integrate.")
@click.option("--tag", "-t", default=None, help="Optional tag for the resulting array.")
@click.option("--label", "-l", help="Custom label for the result.")
@click.pass_context
def magsq(ctx, **kwargs):
  """Calculate the magnitude squared of an input array."""
  verb_print(ctx, "Starting magnitude squared computation")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      postgkyl.tools.mag_sq(dat, output=out)
      data.add(out)
    else:
      postgkyl.tools.mag_sq(dat, output=dat)
    # end
  # end

  verb_print(ctx, "Finishing magnitude squared computation")
