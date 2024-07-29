import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools as tools


@click.command()
@click.argument("axis", nargs=1, type=click.STRING)
@click.option("--use", "-u", default=None, help="Specify the tag to integrate.")
@click.option("--tag", "-t", help="Optional tag for the resulting array.")
@click.option("--label", "-l", help="Custom label for the result.")
@click.pass_context
def integrate(ctx, **kwargs):
  """"Integrate data over a specified axis or axes."""
  verb_print(ctx, "Starting integrate")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    if kwargs["tag"]:
      grid, values = tools.integrate(dat, kwargs["axis"])
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      out.push(grid, values)
      data.add(out)
    else:
      tools.integrate(dat, kwargs["axis"], overwrite=True)
    # end
  # end

  verb_print(ctx, "Finishing integrate")
