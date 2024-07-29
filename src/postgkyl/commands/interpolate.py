import click

from postgkyl.data import GData
from postgkyl.data import GInterpModal, GInterpNodal
from postgkyl.utils import verb_print


@click.command()
@click.option("--basis_type","-b",
    type=click.Choice(["ms", "ns", "mo", "mt", "gkhyb", "pkpmhyb"]),
    help="Specify DG basis.")
@click.option("--poly_order", "-p", type=click.INT, help="Specify polynomial order.")
@click.option("--interp", "-i", type=click.INT,
     help="Interpolation onto a general mesh of specified amount.")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.option("--read", "-r", type=click.BOOL, help="Read from general interpolation file.")
@click.pass_context
def interpolate(ctx, **kwargs):
  """Interpolate DG data onto a uniform mesh."""
  verb_print(ctx, "Starting interpolate")
  data = ctx.obj["data"]

  basis_type = None
  is_modal = None
  if kwargs["basis_type"] is not None:
    if kwargs["basis_type"] == "ms":
      basis_type = "serendipity"
      is_modal = True
    elif kwargs["basis_type"] == "ns":
      basis_type = "serendipity"
      is_modal = False
    elif kwargs["basis_type"] == "mo":
      basis_type = "maximal-order"
      is_modal = True
    elif kwargs["basis_type"] == "mt":
      basis_type = "tensor"
      is_modal = True
    elif kwargs["basis_type"] == "gkhyb":
      basis_type = "gkhybrid"
      is_modal = True
    elif kwargs["basis_type"] == "pkpmhyb":
      basis_type = "hybrid"
      is_modal = True
    # end
  # end

  for dat in data.iterator(kwargs["use"]):
    if kwargs["basis_type"] is None and dat.ctx["basis_type"] is None:
      ctx.fail(
          click.style(f"ERROR in interpolate: no 'basis_type' was specified and dataset {dat.get_label():s} does not have required ctxdata",
              fg="red")
      )
    # end

    if is_modal or dat.ctx["is_modal"]:
      dg = GInterpModal(dat, kwargs["poly_order"], kwargs["basis_type"],
          kwargs["interp"], kwargs["read"])
    else:
      dg = GInterpNodal(dat, kwargs["poly_order"], basis_type, kwargs["interp"], kwargs["read"])
    # end

    num_nodes = dg.num_nodes
    num_comps = int(dat.get_num_comps() / num_nodes)

    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      grid, values = dg.interpolate(tuple(range(num_comps)))
      out.push(grid, values)
      data.add(out)
    else:
      dg.interpolate(tuple(range(num_comps)), overwrite=True)
    # end
  # end
  verb_print(ctx, "Finishing interpolate")
