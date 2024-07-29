import click
import numpy as np

from postgkyl.data import GData
from postgkyl.utils import verb_print


def _get_range(str_in, length):
  if len(str_in.split(",")) > 1:
    return np.array(str_in.split(","), np.int)
  elif str_in.find(":") >= 0:
    str_split = str_in.split(":")

    if str_split[0] == "":
      s_idx = 0
    else:
      s_idx = int(str_split[0])
      if s_idx < 0:
        s_idx = length + s_idx
      # end
    # end

    if str_split[1] == "":
      e_idx = length
    else:
      e_idx = int(str_split[1])
      if e_idx < 0:
        e_idx = length + e_idx
      # end
    # end

    inc = 1
    if len(str_split) > 2 and str_split[2] != "":
      inc = int(str_split[2])
    # end
    return np.arange(s_idx, e_idx, inc)
  else:
    return np.array([int(str_in)])
  # end


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--tag", "-t", help="Tag for the result.")
@click.option("--label", "-l", help="Custom label for the result.")
@click.option("-x", type=click.STRING,
    help="Select components that will became the grid of the new dataset.")
@click.option("-y", type=click.STRING,
    help="Select components that will became the values of the new dataset.")
@click.option("--periodic", "-p", is_flag=True, help="Set the last component to match the first one.")
@click.pass_context
def val2coord(ctx, **kwargs):
  """Given a dataset (typically a DynVector) selects columns from it to create new datasets.

  For example, you can choose say column 1 to be the X-axis of the new dataset and
  column 2 to be the Y-axis. Multiple columns can be choosen using range specifiers and
  as many datasets are then created.
  """
  verb_print(ctx, "Starting val2coord")
  data = ctx.obj["data"]

  tags = list(data.tag_iterator())
  out_tag = kwargs["tag"]
  if out_tag is None:
    if len(tags) == 1:
      out_tag = tags[0]
    else:
      out_tag = "val2coord"
    # end
  # end

  for _, dat in data.iterator(kwargs["use"], enum=True):
    values = dat.get_values()
    x_comps = _get_range(kwargs["x"], len(values[0, :]))
    y_comps = _get_range(kwargs["y"], len(values[0, :]))

    if len(x_comps) > 1 and len(x_comps) != len(y_comps):
      click.echo(
          click.style(f"ERROR 'val2coord': Length of the x-components ({len(x_comps):d}) is greater than 1 and not equal to the y-components ({len(y_comps):d}).",
              fg="red")
      )
      ctx.exit()
    # end

    for i, yc in enumerate(y_comps):
      if len(x_comps) > 1:
        xc = x_comps[i]
      else:
        xc = x_comps[0]
      # end

      x = values[..., xc]
      y = values[..., yc]

      if kwargs["periodic"]:
        x = np.append(x, np.atleast_1d(x[0]), axis=0)
        y = np.append(y, np.atleast_1d(y[0]), axis=0)
      # end

      y = y[..., np.newaxis]  # Adding the required component index

      out = GData(tag=out_tag, label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
      out.push([x], y)
      out.color = "C0"
      data.add(out)
    # end
    dat.deactivate()
  # end
  verb_print(ctx, "Finishing val2coord")
