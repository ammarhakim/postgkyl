import click

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.tools.perprotate


@click.command()
@click.option("--array", "-a", default="array", show_default=True,
    help="Tag for array to be rotated.")
@click.option("--field", "-r", default="field", show_default=True,
    help="Tag for EM field data (data used for the rotation).")
@click.option("--tag", "-t", default="arrayBperp", show_default=True,
    help="Tag for the resulting rotated array perpendicular to magnetic field.")
@click.option("--label", "-l", default="arrayBperp", show_default=True,
    help="Custom label for the result.")
@click.pass_context
def bperprotate(ctx, **kwargs):
  """Rotate an array perpendicular to the unit vectors of the magnetic field.

  For two arrays u and b, where b is the unit vector in the direction of the magnetic
  field, the operation is u - (u dot b_hat) b_hat.
  """
  verb_print(ctx, "Starting rotation perpendicular to magnetic field")

  data = ctx.obj["data"]  # shortcut

  for a, rot in zip(data.iterator(kwargs["array"]), data.iterator(kwargs["field"])):
    # Magnetic field is components 3, 4, & 5 in field array
    grid, outrot = postgkyl.tools.perprotate(a, rot, "3:6")
    # Create new GData structure with appropriate outtag and labels to store output.
    out = GData(tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
        label=kwargs["label"], ctx=a.ctx)
    out.push(grid, outrot)
    data.add(out)
  # end

  data.deactivate_all(tag=kwargs["array"])
  data.deactivate_all(tag=kwargs["field"])

  verb_print(ctx, "Finishing rotation perpendicular to magnetic field")
