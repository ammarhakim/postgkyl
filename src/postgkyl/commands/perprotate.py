import click

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools.perprotate


@click.command()
@click.option("--array", "-a", default="array", show_default=True,
    help="Tag for array to be rotated")
@click.option("--rotator", "-r", default="rotator", show_default=True,
    help="Tag for rotator (data used for the rotation)")
@click.option("--tag", "-t", default="rotarrayperp", show_default=True,
    help="Tag for the resulting rotated array perpendicular to rotator")
@click.option("--label", "-l", default="rotarrayperp", show_default=True,
    help="Custom label for the result")
@click.pass_context
def perprotate(ctx, **kwargs):
  """Rotate an array perpendicular to the unit vectors of a second array.

  For two arrays u and v, where v is the rotator, operation is u - (u dot v_hat) v_hat.
  """
  verb_print(ctx, "Starting rotation perpendicular to rotator array")

  data = ctx.obj["data"]  # shortcut

  for a, rot in zip(data.iterator(kwargs["array"]), data.iterator(kwargs["rotator"])):
    grid, outrot = postgkyl.tools.perprotate(a, rot)
    # Create new GData structure with appropriate outtag and labels to store output.
    out = GData(tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
        label=kwargs["label"], ctx=a.ctx)
    out.push(outrot, grid)
    data.add(out)
  # end

  data.deactivate_all(tag=kwargs["array"])
  data.deactivate_all(tag=kwargs["rotator"])

  verb_print(ctx, "Finishing rotation perpendicular to rotator array")
