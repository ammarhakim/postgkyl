import click

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools.parrotate


@click.command()
@click.option("--array", "-a", default="array", show_default=True,
     help="Tag for array to be rotated")
@click.option("--rotator", "-r", default="rotator", show_default=True,
    help="Tag for rotator (data used for the rotation)")
@click.option("--tag", "-t", default="rotarraypar", show_default=True,
    help="Tag for the resulting rotated array parallel to rotator")
@click.option("--label", "-l", default="rotarraypar", show_default=True,
    help="Custom label for the result")
@click.pass_context
def parrotate(ctx, **kwargs):
  """Rotate an array parallel to the unit vectors of a second array.

  For two arrays u and v, where v is the rotator, operation is (u dot v_hat) v_hat. Note
  that for a three-component field, the output is a new vector whose components are
  (u_{v_x}, u_{v_y}, u_{v_z}), i.e., the x, y, and z components of the vector u parallel
  to v.
  """
  verb_print(ctx, "Starting rotation parallel to rotator array")

  data = ctx.obj["data"]

  for a, rot in zip(data.iterator(kwargs["array"]), data.iterator(kwargs["rotator"])):
    grid, outrot = postgkyl.tools.parrotate(a, rot)
    # Create new GData structure with appropriate outtag and labels to store output.
    out = GData(tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
        label=kwargs["label"], ctx=a.ctx)
    out.push(outrot, grid)
    data.add(out)
  # end

  data.deactivate_all(tag=kwargs["array"])
  data.deactivate_all(tag=kwargs["rotator"])

  verb_print(ctx, "Finishing rotation parallel to rotator array")
