import click
import numpy as np

from postgkyl.data import GData
from postgkyl.utils import verb_print


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--tag", "-t", type=click.STRING, help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.option("--read", "-r", type=click.BOOL, help="Read from general interpolation file.")
@click.pass_context
def grid(ctx, **kwargs):
  """Create a dataset out of a grid"""
  verb_print(ctx, "Starting grid")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    grid_in = dat.get_grid()
    num_dims = dat.get_num_dims()
    num_cells = dat.get_num_cells()
    grid_out = []
    for nc in num_cells:
      grid_out.append(np.arange(nc+1))
    # end

    shape = np.copy(num_cells) + 1
    shape = np.append(shape, num_dims)
    values = np.zeros(shape)

    if num_dims == 1:
      values[..., 0] = grid_in[0]
    elif len(grid_in[-1].shape) == 1: # uniform mesh
      temp = np.meshgrid(*grid_in, indexing="ij")
      for d, t in enumerate(temp):
        values[..., d] = t
      # end
    else: # c2p mapping
      for d, t in enumerate(grid_in):
        values[..., d] = t
      # end
    # end

    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      out.push(grid_out, values)
      data.add(out)
    else:
      dat.push(grid_out, values)
    # end
  # end
  verb_print(ctx, "Finishing grid")
