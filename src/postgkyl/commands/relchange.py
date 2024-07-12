import click

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools as diag


@click.command(help="Computes the relative change between two datasets")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option(
    "--index",
    "-i",
    default=0,
    show_default=True,
    help="Dataset index for computing change relative to",
)
@click.option(
    "--comp",
    "-c",
    default=None,
    show_default=True,
    help="Dataset component to be compared to \
              if user only wants to compare to a single component",
)
@click.option(
    "--tag", "-t", default="relChange", show_default=True, help="Tag for the result"
)
@click.option(
    "--label",
    "-l",
    default="delta",
    show_default=True,
    help="Custom label for the result",
)
@click.pass_context
def relchange(ctx, **kwargs):
  verb_print(ctx, "Starting relative change")

  data = ctx.obj["data"]  # shortcut
  for tag in data.tag_iterator(kwargs["use"]):
    reference = data.get_dataset(tag, kwargs["index"])
    for dat in data.iterator(tag):
      if kwargs["tag"]:
        out = GData(tag=kwargs["tag"], compgrid=ctx.obj["compgrid"], ctx=dat.ctx)
        grid, values = diag.rel_change(reference, dat, kwargs["comp"])
        dat.deactivate()
        out.push(grid, values)
        data.add(out)
      else:
        diag.rel_change(reference, dat, kwargs["comp"], overwrite=True)
      # end
    # end
  # end
  verb_print(ctx, "Finishing relative change")
