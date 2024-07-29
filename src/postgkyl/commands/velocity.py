import click

from postgkyl.data import GData
from postgkyl.utils import verb_print


@click.command()
@click.option("--density", "-d", default="density", show_default=True, help="Tag for density.")
@click.option("--momentum", "-m", default="momentum", show_default=True, help="Tag for momentum.")
@click.option("--tag", "-t", default="velocity", show_default=True, help="Tag for the result.")
@click.option("--label", "-l", default="velocity", show_default=True,
    help="Custom label for the result.")
@click.pass_context
def velocity(ctx, **kwargs):
  verb_print(ctx, "Starting velocity")

  data = ctx.obj["data"]  # shortcut

  for m0, m1 in zip(data.iterator(kwargs["density"]), data.iterator(kwargs["momentum"])):
    grid = m0.get_grid()
    vals_M0 = m0.get_values()
    vals_M1 = m1.get_values()

    out = GData(tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
        label=kwargs["label"], ctx=m0.ctx)
    out.push(grid, vals_M1 / vals_M0)
    data.add(out)
  # end

  data.deactivate_all(tag=kwargs["density"])
  data.deactivate_all(tag=kwargs["momentum"])

  verb_print(ctx, "Finishing velocity")
