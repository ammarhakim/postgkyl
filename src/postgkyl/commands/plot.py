import click
import matplotlib.pyplot as plt
import numpy as np

from postgkyl.utils import verb_print
import postgkyl.output.plot


@click.command()
@click.option("--use", "-u", default=None, help="Specify the tag to plot.")
@click.option("--figure", "-f", default=None,
    help="Specify figure to plot in; either number or 'dataset'.")
@click.option("--squeeze", is_flag=True, help="Squeeze the components into one panel.")
@click.option("--subplots", "-b", is_flag=True, help="Make subplots from multiple datasets.")
@click.option("--nsubplotrow", "num_subplot_row", type=click.INT,
    help="Manually set the number of rows for subplots.")
@click.option("--nsubplotcol", "num_subplot_col", type=click.INT,
    help="Manually set the number of columns for subplots.")
@click.option("--transpose", is_flag=True, help="Transpose axes.")
@click.option("-c", "--contour", is_flag=True, help="Make contour plot.")
@click.option("--clevels", type=click.STRING,
    help="Specify levels for contours: comma-separated level values or start:end:nlevels.")
@click.option("--cnlevels", type=click.INT, help="Specify the number of levels for contours.")
@click.option("--contlabel", "cont_label", is_flag=True, help="Add labels to contours")
@click.option("-q", "--quiver", is_flag=True, help="Make quiver plot.")
@click.option("-l", "--streamline", is_flag=True, help="Make streamline plot.")
@click.option("--sdensity", type=click.INT, default=1, help="Control density of the streamlines.")
@click.option("--arrowstyle", type=click.STRING, help="Set the style for streamline arrows.")
@click.option("--lineouts", type=click.Choice(["0", "1"]), help="Switch to lineouts mode.")
@click.option("-s", "--scatter", is_flag=True, help="Make scatter plot.")
@click.option("--markersize", type=click.FLOAT, help="Set marker size for scatter plots.")
@click.option("--linewidth", type=click.FLOAT, help="Set the linewidth.")
@click.option("--linestyle", type=click.Choice(["solid", "dashed", "dotted", "dashdot"]),
    help="Set the linestyle.")
@click.option("--style", help="Specify Matplotlib style file (default: Postgkyl).")
@click.option("-d", "--diverging", is_flag=True, help="Switch to diverging color map.")
@click.option("--arg", type=click.STRING, default="",
    help="Additional plotting arguments, e.g., '*--'.")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Enforce the same scaling on both axes.")
@click.option("--aspect", default=None, help="Specify the scaling ratio.")
@click.option("--logx", is_flag=True, help="Set x-axis to log scale.")
@click.option("--logy", is_flag=True, help="Set y-axis to log scale.")
@click.option("--logz", is_flag=True, help="Set values of 2D plot to log scale.")
@click.option("--xshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the x-axis.")
@click.option("--yshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the y-axis.")
@click.option("--zshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the z-axis.")
@click.option("--xscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the x-axis.")
@click.option("--yscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the y-axis.")
@click.option("--zscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the z-axis (default: 1.0).")
@click.option("--xmax", default=None, type=click.FLOAT, help="Set maximal x-value.")
@click.option("--xmin", default=None, type=click.FLOAT, help="Set minimal x-values.")
@click.option("--ymax", default=None, type=click.FLOAT, help="Set maximal y-value.")
@click.option("--ymin", default=None, type=click.FLOAT, help="Set minimal y-values.")
@click.option("--zmax", default=None, type=click.FLOAT, help="Set maximal z-value.")
@click.option("--zmin", default=None, type=click.FLOAT, help="Set minimal z-values.")
@click.option("--xlim", default=None, type=click.STRING,
    help="Set limits for the x-coordinate (lower,upper)")
@click.option("--ylim", default=None, type=click.STRING,
    help="Set limits for the y-coordinate (lower,upper).")
@click.option("--zlim", default=None, type=click.STRING,
    help="Set limits for the z-coordinate (lower,upper).")
@click.option("--relax", is_flag=True, help="Relax the stringent x axis limits for 1D plots.")
@click.option("--globalrange", "-r", is_flag=True, help="Make uniform extends across datasets.")
@click.option("--legend/--no-legend", default=True, help="Show legend.")
@click.option("--force-legend", "forcelegend", is_flag=True,
    help="Force legend even when plotting a single dataset.")
@click.option("--color", type=click.STRING, help="Set color when available.")
@click.option("-x", "--xlabel", type=click.STRING, help="Specify a x-axis label.")
@click.option("-y", "--ylabel", type=click.STRING, help="Specify a y-axis label.")
@click.option("--clabel", type=click.STRING, help="Specify a label for colorbar.")
@click.option("--title", type=click.STRING, help="Specify a title.")
@click.option("--save", is_flag=True, help="Save figure as PNG file.")
@click.option("--saveas", type=click.STRING, default=None, help="Name of figure file.")
@click.option("--dpi", type=click.INT, default=200, help="DPI (resolution) for output.")
@click.option("-e", "--edgecolors", type=click.STRING,
    help="Set color for cell edges to show grid outline.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show grid-lines.")
@click.option("--xkcd", is_flag=True, help="Turns on the xkcd style!")
@click.option("--hashtag", is_flag=True, help="Turns on the pgkyl hashtag!")
@click.option("--show/--no-show", default=True,
    help="Turn showing of the plot ON and OFF.")
@click.option("--figsize", help="Comma-separated values for x and y size.")
@click.option("--saveframes", type=click.STRING,
    help="Save individual frames as PNGS instead of an opening them")
@click.option("--jet", is_flag=True, help="Turn colormap to jet for comparison with literature.")
@click.option("--cmap", type=click.STRING, default=None,
    help="Override default colormap with a valid matplotlib cmap.")
@click.pass_context
def plot(ctx, **kwargs):
  """Plot active datasets, optionally displaying the plot and/or saving it to PNG files.

  Plot labels can use a sub-set of LaTeX math commands placed between dollar ($) signs.
  """
  verb_print(ctx, "Starting plot")

  kwargs["rcParams"] = ctx.obj["rcParams"]

  args = kwargs["arg"]
  if kwargs["scatter"]:
    args += "."
  # end
  del kwargs["arg"]

  if kwargs["jet"]:
    click.echo(
        click.style("WARNING: The 'jet' colormap has been selected. This colormap is not perceptually uniform and seemingly creates features which do not exist in the data!",
            fg="yellow")
    )
  # end

  if kwargs["aspect"]:
    kwargs["fixaspect"] = True
  # end

  if kwargs["lineouts"]:
    kwargs["lineouts"] = int(kwargs["lineouts"])
  # end

  kwargs["num_axes"] = None
  if kwargs["subplots"]:
    kwargs["num_axes"] = 0
    kwargs["start_axes"] = 0
    for dat in ctx.obj["data"].iterator(kwargs["use"]):
      kwargs["num_axes"] = kwargs["num_axes"] + dat.get_num_comps()
    # end
    if kwargs["figure"] is None:
      kwargs["figure"] = 0
    # end
  # end

  if kwargs["xlim"]:
    kwargs["xmin"] = float(kwargs["xlim"].split(",")[0])
    kwargs["xmax"] = float(kwargs["xlim"].split(",")[1])
  # end
  if kwargs["ylim"]:
    kwargs["ymin"] = float(kwargs["ylim"].split(",")[0])
    kwargs["ymax"] = float(kwargs["ylim"].split(",")[1])
  # end
  if kwargs["zlim"]:
    kwargs["zmin"] = float(kwargs["zlim"].split(",")[0])
    kwargs["zmax"] = float(kwargs["zlim"].split(",")[1])
  # end

  dataset_fignum = False
  if (
      kwargs["figure"] == "dataset"
      or kwargs["figure"] == "set"
      or kwargs["figure"] == "s"
  ):
    dataset_fignum = True
  # end

  if kwargs["globalrange"]:
    vmin = float("inf")
    vmax = float("-inf")
    for dat in ctx.obj["data"].iterator(kwargs["use"]):
      val = dat.get_values() * kwargs["zscale"]
      if vmin > np.nanmin(val):
        vmin = np.nanmin(val)
      # end
      if vmax < np.nanmax(val):
        vmax = np.nanmax(val)
      # end
    # end

    if kwargs["zmin"] is None:
      kwargs["zmin"] = vmin
    # end
    if kwargs["zmax"] is None:
      kwargs["zmax"] = vmax
    # end
  # end

  file_name = ""

  # ---- Loop over all the datasets ----
  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    if dataset_fignum:
      kwargs["figure"] = int(i)
    # end
    if ctx.obj["data"].get_num_datasets() > 1 or kwargs["forcelegend"]:
      label = dat.get_label()
    else:
      label = ""
    # end

    # ---- Plot ----
    postgkyl.output.plot(dat, args, label_prefix=label, **kwargs)

    if kwargs["subplots"]:
      kwargs["start_axes"] = kwargs["start_axes"] + dat.get_num_comps()
    # end

    if kwargs["save"] or kwargs["saveas"]:
      if kwargs["saveas"]:
        file_name = kwargs["saveas"]
      else:
        if file_name != "":
          file_name = file_name + "_"
        # end
        if dat._file_name:
          file_name = file_name + dat._file_name.split(".")[0]
        else:
          file_name = file_name + "ev_" + ctx.obj["labels"][i].replace(" ", "_")
        # end
      # end
    # end
    if (kwargs["save"] or kwargs["saveas"]) and kwargs["figure"] is None:
      file_name = str(file_name)
      plt.savefig(file_name, dpi=kwargs["dpi"])
      file_name = ""
    # end

    if kwargs["saveframes"]:
      file_name = f"{kwargs['saveframes']:s}_{i:d}.png"
      plt.savefig(file_name, dpi=kwargs["dpi"])
      kwargs["show"] = False
    # end
  # end
  if (kwargs["save"] or kwargs["saveas"]) and kwargs["figure"] is not None:
    file_name = str(file_name)
    plt.savefig(file_name, dpi=kwargs["dpi"])
  # end

  if kwargs["show"]:
    plt.show()
  # end
  verb_print(ctx, "Finishing plot")
