import click
import numpy as np

from postgkyl.data import GInterpModal
from postgkyl.utils import verb_print

from postgkyl.data import GData


@click.command(help="Interpolate DG data on a uniform mesh")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.option(
    "--basis_type", "-b", type=click.Choice(["ms", "ns", "mo"]), help="Specify DG basis"
)
@click.option("--poly_order", "-p", type=click.INT, help="Specify polynomial order")
@click.option("--interp", "-i", type=click.INT, help="Number of poins to evaluate on")
@click.option(
    "-r", "--periodic", is_flag=True, help="Flag for periodic boundary conditions"
)
@click.option("-c", "--c1", is_flag=True, help="Enforce continuous first derivatives")
@click.pass_context
def recovery(ctx, **kwargs):
  verb_print(ctx, "Starting recovery")
  data = ctx.obj["data"]

  if kwargs["basis_type"] is not None:
    if kwargs["basis_type"] == "ms" or kwargs["basis_type"] == "ns":
      basis_type = "serendipity"
    elif kwargs["basis_type"] == "mo":
      basis_type = "maximal-order"
    # end
  else:
    basis_type = None
  # end

  for dat in data.iterator(kwargs["use"]):
    dg = GInterpModal(
        dat, kwargs["poly_order"], basis_type, kwargs["interp"], kwargs["periodic"]
    )
    num_nodes = dg.num_nodes
    num_comps = int(dat.get_num_comps() / num_nodes)

    # verb_print(ctx, 'interplolate: interpolating dataset #{:d}'.format(s))
    # dg.recovery(tuple(range(num_comps)), stack=True)
    if kwargs["tag"]:
      out = GData(
          tag=kwargs["tag"],
          label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"],
          ctx=dat.ctx,
      )
      grid, values = dg.recovery(0, kwargs["c1"])
      out.push(grid, values)
      data.add(out)
    else:
      dg.recovery(0, kwargs["c1"], overwrite=True)
    # end
  # end
  verb_print(ctx, "Finishing recovery")


# end
