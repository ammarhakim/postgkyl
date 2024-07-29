import click
import numpy as np

from postgkyl.data import GData
from postgkyl.utils import verb_print

import postgkyl.tools.energetics


@click.command()
@click.option("--elc", "-e", default="elc", show_default=True, help="Tag for electrons.")
@click.option("--ion", "-i", default="ion", show_default=True, help="Tag for ions.")
@click.option("--field", "-f", default="field", show_default=True, help="Tag for EM fields.")
@click.option("--tag", "-t", default="energetics", show_default=True, help="Tag for the result.")
@click.option("--label", "-l", default="E", show_default=True, help="Custom label for the result.")
@click.pass_context
def energetics(ctx, **kwargs):
  """Decomposes the components of the energy (kinetic, thermal, electromagnetic) for a two-species (electron, ion) plasma."""
  verb_print(ctx, "Starting energetics decomposition")
  data = ctx.obj["data"]  # shortcut

  for elc, ion, em in zip(data.iterator(kwargs["elc"]),
      data.iterator(kwargs["ion"]), data.iterator(kwargs["field"])):
    grid = em.get_grid()
    out_energetics = np.zeros(em.get_values()[..., 0:7].shape)
    out = GData(tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
        label=kwargs["label"], ctx=em.ctx)
    grid, out_energetics = postgkyl.tools.energetics(elc, ion, em)
    out.push(grid, out_energetics)
    data.add(out)
  # end

  data.deactivate_all(tag=kwargs["elc"])
  data.deactivate_all(tag=kwargs["ion"])
  data.deactivate_all(tag=kwargs["field"])

  verb_print(ctx, "Finishing energetics decomposition")
