import click
import glob

from postgkyl.data import GData
from postgkyl.data import GInterpModal
from postgkyl.utils import verb_print


def _pick_cut(ctx, kwargs, zn):
  nm = f"z{zn:d}"
  if zn == 6:  # This little hack allows to apply the same function for
    # components as well
    nm = "component"
  # end
  if kwargs[nm] and ctx.obj["global_cuts"][zn]:
    click.echo(click.style(f"WARNING: The local '{nm:s}' is overwriting the global '{nm:s}'",
        fg="yellow"))
    return kwargs[nm]
  elif kwargs[nm]:
    return kwargs[nm]
  elif ctx.obj["global_cuts"][zn]:
    return ctx.obj["global_cuts"][zn]
  else:
    return None
  # end


def _crush(s):  # Temp function used as a sorting key
  splitted = s.split("_")
  tmp = splitted[-1].split(".")
  splitted[-1] = int(tmp[0])
  splitted.append(tmp[1])
  return tuple(splitted)


@click.command(hidden=True)
@click.option("--z0", help="Partial file load: 0th coord (either int or slice).")
@click.option("--z1", help="Partial file load: 1st coord (either int or slice).")
@click.option("--z2", help="Partial file load: 2nd coord (either int or slice).")
@click.option("--z3", help="Partial file load: 3rd coord (either int or slice).")
@click.option("--z4", help="Partial file load: 4th coord (either int or slice).")
@click.option("--z5", help="Partial file load: 5th coord (either int or slice).")
@click.option("--component", "-c", help="Partial file load: comps (either int or slice).")
@click.option("--tag", "-t", default="default", help="Specily tag for data.")
@click.option("--compgrid", is_flag=True, help="Disregard the mapped grid information")
@click.option("--varname", "-d", multiple=True,
    help="Allows to specify the Adios variable name. [default: 'CartGridField']")
@click.option("--label", "-l", help="Allows to specify the custom label")
@click.option("--c2p", type=click.STRING,
    help="Specify the file name containing c2p mapped coordinates")
@click.option("--c2p-vel", "c2p_vel",type=click.STRING,
    help="Specify the file name containing c2p mapped coordinates")
@click.option("--fv", is_flag=True,
    help="Tag finite volume data when using c2p mapped coordinates")
@click.option("--reader", "-r", type=click.STRING,
    help="Allows to specify the Adios variable name (default is 'CartGridField')")
@click.option("--load/--no-load", default=True, help="Specify if data should be loaded.")
@click.pass_context
def load(ctx, **kwargs):
  verb_print(ctx, "Starting load")
  data = ctx.obj["data"]

  idx = ctx.obj["in_data_strings_loaded"]
  in_data_string = ctx.obj["in_data_strings"][idx]

  # Handling the wildcard characters
  if "*" in in_data_string or "?" in in_data_string or "!" in in_data_string:
    files = glob.glob(str(in_data_string))
    files = [f for f in files if f.find("restart") < 0]
    try:
      files = sorted(files, key=_crush)
    except Exception:
      click.echo(
          click.style("WARNING: The loaded files appear to be of different types. Sorting is turned off.",
              fg="yellow")
      )
    # end
  else:
    files = [in_data_string]
  # end

  # Resolve the local/global variable names and partial loading
  # The local settings take a precedents but a warning is going to appear
  z0 = _pick_cut(ctx, kwargs, 0)
  z1 = _pick_cut(ctx, kwargs, 1)
  z2 = _pick_cut(ctx, kwargs, 2)
  z3 = _pick_cut(ctx, kwargs, 3)
  z4 = _pick_cut(ctx, kwargs, 4)
  z5 = _pick_cut(ctx, kwargs, 5)
  comp = _pick_cut(ctx, kwargs, 6)

  var_names = ["CartGridField"]
  if kwargs["varname"] and ctx.obj["global_var_names"]:
    var_names = kwargs["varname"]
    click.echo(
        click.style("WARNING: The local 'varname' is overwriting the global 'varname'",
            fg="yellow")
    )
  elif kwargs["varname"]:
    var_names = kwargs["varname"]
  elif ctx.obj["global_var_names"]:
    var_names = ctx.obj["global_var_names"]
  # end

  mapc2p_name = None
  if kwargs["c2p"] and ctx.obj["global_c2p"]:
    mapc2p_name = kwargs["c2p"]
    click.echo(
        click.style("WARNING: The local 'c2p' is overwriting the global 'c2p'", fg="yellow")
    )
  elif kwargs["c2p"]:
    mapc2p_name = kwargs["c2p"]
  elif ctx.obj["global_c2p"]:
    mapc2p_name = ctx.obj["global_c2p"]
  # end

  mapc2p_vel_name = None
  if kwargs["c2p_vel"] and ctx.obj["global_c2p_vel"]:
    mapc2p_name = kwargs["c2p_vel"]
    click.echo(
        click.style("WARNING: The local 'c2p_vel' is overwriting the global 'c2p_vel'",
            fg="yellow")
    )
  elif kwargs["c2p_vel"]:
    mapc2p_vel_name = kwargs["c2p_vel"]
  elif ctx.obj["global_c2p_vel"]:
    mapc2p_vel_name = ctx.obj["global_c2p_vel"]
  # end

  if len(var_names) == 1:
    var_names = var_names[0].split(",")
  # end

  for var in var_names:
    for fn in files:
      try:
        dat = GData(file_name=fn, tag=kwargs["tag"], comp_grid=ctx.obj["compgrid"],
            z0=z0, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5, comp=comp, var_name=var,
            label=kwargs["label"], mapc2p_name=mapc2p_name, mapc2p_vel_name=mapc2p_vel_name,
            reader_name=kwargs["reader"], load=kwargs["load"], click_mode=True)
        if kwargs["fv"]:
          dg = GInterpModal(dat, 0, "ms")
          dg.interpolateGrid(overwrite=True)
        # end
        data.add(dat)
      except NameError as e:
        ctx.fail(click.style(rf"{repr(e):s}", fg="red"))
      # end
    # end
  # end

  data.set_unique_labels()

  ctx.obj["in_data_strings_loaded"] += 1
  verb_print(ctx, "Finishing load")
