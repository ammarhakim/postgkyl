import click
import importlib
import numpy as np
import os.path
from pathlib import Path
import webbrowser

from postgkyl.utils import verb_print


def _parse_range_option(_ctx, _param, value):
  if value is None:
    return None
  # end

  parts = [part.strip() for part in str(value).replace(":", ",").split(",") if part.strip()]
  if len(parts) != 2:
    raise click.BadParameter("Expected two numbers in the form 'lower,upper' or 'lower:upper'.")
  # end

  try:
    return (float(parts[0]), float(parts[1]))
  except ValueError as exc:
    raise click.BadParameter("Expected two numbers in the form 'lower,upper' or 'lower:upper'.") from exc
  # end


def _parse_slice_option(_ctx, _param, value):
  if value is None:
    return None
  # end

  tokens = [token.strip() for token in str(value).split(",") if token.strip()]
  if not tokens:
    raise click.BadParameter("Expected a number or comma-separated list of numbers.")
  # end

  selectors = []
  for token in tokens:
    token_lower = token.lower()
    if "." in token_lower or "e" in token_lower:
      try:
        selectors.append(float(token))
      except ValueError as exc:
        raise click.BadParameter(
            f"Invalid selector '{token}'. Use int for index or float for coordinate value."
        ) from exc
      # end
    else:
      try:
        selectors.append(int(token))
      except ValueError as exc:
        raise click.BadParameter(
            f"Invalid selector '{token}'. Use int for index or float for coordinate value."
        ) from exc
      # end
    # end
  # end
  return selectors


@click.command(name="plot3d")
@click.option("--use", "-u", default=None, help="Specify the tag to plot.")
@click.option("--figure", "-f", default=None,
    help="Specify figure to plot in; either number or 'dataset'.")
@click.option("--squeeze", is_flag=True, help="Squeeze the components into one panel.")
@click.option("--subplots", "-b", is_flag=True, help="Make subplots from multiple datasets.")
@click.option("--nsubplotrow", "num_subplot_row", type=click.INT,
    help="Manually set the number of rows for subplots.")
@click.option("--nsubplotcol", "num_subplot_col", type=click.INT,
    help="Manually set the number of columns for subplots.")
@click.option("-q", "--quiver", is_flag=True, help="Make quiver plot.")
@click.option("-l", "--streamline", is_flag=True, help="Make streamline plot.")
@click.option("--sdensity", type=click.INT, default=1, help="Control density of the streamlines.")
@click.option("-o", "--opacity", type=click.FLOAT, default=1.0, show_default=True,
    help="Set opacity for 3D volume plots (0.0-1.0).")
@click.option("--surface-count", type=click.INT, default=32, show_default=True,
    help="Number of Plotly volume isosurfaces to render for 3D plots.")
@click.option("--maximum-points-per-axis", "--mppa", "maximum_points_per_axis", type=click.INT, default=0, show_default=True,
    help="Maximum number of points along any 3D volume axis; 0 disables downsampling.")
@click.option("--style", help="Specify Matplotlib style file (default: Postgkyl).")
@click.option("--background", type=click.Choice(["dark", "light"]), default="dark", show_default=True,
    help="Background mode for plots (dark/light).")
@click.option("-d", "--diverging", is_flag=True, help="Switch to diverging color map.")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Enforce the same scaling on all 3D axes.")
@click.option("--aspect", default=None,
    help="Specify aspect behavior: auto,data,cube, or numeric ratio.")
@click.option("--logx", is_flag=True, help="Set x-axis to log scale.")
@click.option("--logy", is_flag=True, help="Set y-axis to log scale.")
@click.option("--logz", is_flag=True, help="Set z-axis to log scale.")
@click.option("--logc", is_flag=True, help="Set colorbar to log scale.")
@click.option("--xshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the x-axis.")
@click.option("--yshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the y-axis.")
@click.option("--zshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the z-axis.")
@click.option("--cshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Value to shift the color values.")
@click.option("--xscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the x-axis.")
@click.option("--yscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the y-axis.")
@click.option("--zscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the z-axis.")
@click.option("--cscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Value to scale the color values.")
@click.option("--slice-at-z0", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z0 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--slice-at-z1", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z1 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--slice-at-z2", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z2 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--slice-at-z3", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z3 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--slice-at-z4", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z4 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--slice-at-z5", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Select z5 slices. Comma-separated selectors; ints are indices, floats are coordinate values.")
@click.option("--xmax", default=None, type=click.FLOAT, help="Set maximal x-value.")
@click.option("--xmin", default=None, type=click.FLOAT, help="Set minimal x-value.")
@click.option("--ymax", default=None, type=click.FLOAT, help="Set maximal y-value.")
@click.option("--ymin", default=None, type=click.FLOAT, help="Set minimal y-value.")
@click.option("--zmax", default=None, type=click.FLOAT, help="Set maximal z-value.")
@click.option("--zmin", default=None, type=click.FLOAT, help="Set minimal z-value.")
@click.option("--cmax", default=None, type=click.FLOAT, help="Set maximal color value.")
@click.option("--cmin", default=None, type=click.FLOAT, help="Set minimal color value.")
@click.option("--xlim", default=None, type=click.STRING, callback=_parse_range_option,
    help="Set limits for the x-coordinate (lower,upper).")
@click.option("--ylim", default=None, type=click.STRING, callback=_parse_range_option,
    help="Set limits for the y-coordinate (lower,upper).")
@click.option("--zlim", default=None, type=click.STRING, callback=_parse_range_option,
    help="Set limits for the z-coordinate (lower,upper).")
@click.option("--clim", default=None, type=click.STRING, callback=_parse_range_option,
    help="Set limits for the color scale (lower,upper).")
@click.option("--globalrange", "-r", is_flag=True, help="Make uniform extents across datasets.")
@click.option("--cutoffglobalrange", "-cogr", default=None, type=click.FLOAT,
    help="Set custom percentile cutoff for uniform ranges.")
@click.option("--legend", default=None, type=click.STRING,
    help="If specified, comma-separated legend labels (e.g., 'a,b,c').")
@click.option("--no-legend", is_flag=True, help="Hide legend.")
@click.option("--force-legend", "forcelegend", is_flag=True,
    help="Force legend even when plotting a single dataset.")
@click.option("--color", type=click.STRING, help="Set color when available.")
@click.option("-x", "--xlabel", type=click.STRING, help="Specify an x-axis label.")
@click.option("-y", "--ylabel", type=click.STRING, help="Specify a y-axis label.")
@click.option("-z", "--zlabel", type=click.STRING, help="Specify a z-axis label.")
@click.option("--clabel", type=click.STRING, help="Specify a label for colorbar.")
@click.option("--title", type=click.STRING, help="Specify a title.")
@click.option("--subplot-titles", type=click.STRING,
    help="Comma-separated titles for each subplot.")
@click.option("--subplot-xlabels", type=click.STRING,
    help="Comma-separated x-axis labels for each subplot.")
@click.option("--subplot-ylabels", type=click.STRING,
    help="Comma-separated y-axis labels for each subplot.")
@click.option("--save", is_flag=True, help="Save plot output.")
@click.option("--saveas", type=click.STRING, default=None, help="Output file name.")
@click.option("--starting-azimuthal-angle", "azimuthal_angle", "--azimuthal-angle",
    type=click.FLOAT, default=0.0, show_default=True,
    help="Starting azimuthal angle in degrees for rotating 3D save.")
@click.option("--polar-angle", type=click.FLOAT, default=85.0, show_default=True,
    help="Polar angle in degrees for rotating 3D camera. 90 degrees is the x-y plane.")
@click.option("--rotation-period", type=click.FLOAT, default=20.0, show_default=True,
    help="Rotation period in seconds for one full rotation (for rotating html/mp4/gif output).")
@click.option("--fps", type=click.INT, default=1, show_default=True,
    help="FPS used for rotating mp4/gif save output.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show grid-lines.")
@click.option("--hashtag", is_flag=True, help="Turns on the pgkyl hashtag!")
@click.option("--show/--no-show", default=True,
    help="Turn showing of the plot ON and OFF.")
@click.option("--figsize", help="Comma-separated values for x and y size.")
@click.option("--saveframes", type=click.STRING,
    help="Save one output per dataset with this prefix.")
@click.option("--jet", is_flag=True, help="Turn colormap to jet for comparison with literature.")
@click.option("--cmap", type=click.STRING, default=None,
    help="Override default colormap with a valid matplotlib cmap.")
@click.option("--invert-cmap", is_flag=True,
    help="Invert the selected colormap (or the default colormap for the chosen background mode).")
@click.option("-m", "--multiblock", is_flag=True, default=False)
@click.pass_context
def plot3d(ctx, **kwargs):
  """Plot active 3D datasets with Plotly and optional rotating export."""
  verb_print(ctx, "Starting plot3d")
  plot_output_module = importlib.import_module("postgkyl.output.plot3d")

  def _save_output_3d(fig, file_name: str | None = None, base_name: str | None = None,
      force_rotating_preview: bool = False) -> str:
    if force_rotating_preview:
      safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (base_name or "")).strip("_")
      if not safe_base:
        safe_base = "plot3d_preview"
      # end
      file_name = os.path.join(os.getcwd(), f"{safe_base}_preview.html")
    elif file_name is None:
      raise click.ClickException("Internal error: missing output file name for 3D save.")
    # end

    root, ext = os.path.splitext(file_name)
    ext = ext.lower()
    rotating_target = force_rotating_preview or ext in (".mp4", ".gif", ".html")
    if rotating_target:
      if ext == "":
        file_name = f"{file_name}.mp4"
      # end
      plot_output_module.save_rotating_plotly_figure(
          fig,
          file_name,
          starting_azimuthal_angle=kwargs["azimuthal_angle"],
          polar_angle=kwargs["polar_angle"],
          rotation_period=kwargs["rotation_period"],
          fps=kwargs["fps"],
      )
      return file_name
    # end

    if ext != ".html":
      file_name = f"{root}.html" if root else f"{file_name}.html"
    # end
    fig.write_html(file_name)
    return file_name

  def _open_html_preview(html_name: str):
    webbrowser.open(Path(html_name).resolve().as_uri())

  kwargs["rcParams"] = ctx.obj["rcParams"]

  if kwargs["jet"]:
    click.echo(
        click.style("WARNING: The 'jet' colormap has been selected. This colormap is not perceptually uniform and seemingly creates features which do not exist in the data!",
            fg="yellow")
    )
  # end

  if kwargs["aspect"]:
    kwargs["fixaspect"] = True
  # end

  slice_kwargs = {}
  for d in range(6):
    slice_selectors = kwargs.pop(f"slice_at_z{d}")
    if slice_selectors is not None:
      slice_kwargs[f"z{d}"] = slice_selectors
    # end
  # end

  def _get_slice_kwargs_for_data(dat):
    if not slice_kwargs:
      return {}
    # end

    num_dims = dat.get_num_dims()
    if num_dims != 3:
      raise click.ClickException("Slice overlays are only supported for 3D datasets in plot3d.")
    # end

    resolved = {}
    for key, selectors in slice_kwargs.items():
      axis = int(key[1:])
      if axis >= num_dims:
        raise click.ClickException(
            f"Cannot use --slice-at-{key} on a {num_dims:d}D dataset."
        )
      # end
      if selectors:
        resolved[key] = selectors
      # end
    # end
    return resolved

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
    kwargs["xmin"], kwargs["xmax"] = kwargs["xlim"]
    kwargs["xrange"] = kwargs["xlim"]
  # end
  if kwargs["ylim"]:
    kwargs["ymin"], kwargs["ymax"] = kwargs["ylim"]
    kwargs["yrange"] = kwargs["ylim"]
  # end
  if kwargs["zlim"]:
    kwargs["zmin"], kwargs["zmax"] = kwargs["zlim"]
    kwargs["zrange"] = kwargs["zlim"]
  # end
  if kwargs["clim"]:
    kwargs["cmin"], kwargs["cmax"] = kwargs["clim"]
  # end

  dataset_fignum = kwargs["figure"] in ("dataset", "set", "s")

  if kwargs["multiblock"] and kwargs["cutoffglobalrange"] is None:
    kwargs["globalrange"] = True
  # end

  if kwargs["globalrange"] or kwargs["cutoffglobalrange"]:
    vmin = float("inf")
    vmax = float("-inf")
    v_extrema = np.array([])
    for dat in ctx.obj["data"].iterator(kwargs["use"]):
      if dat.get_num_dims() != 3:
        continue
      # end
      val = dat.get_values() * kwargs["zscale"]
      if vmin > np.nanmin(val):
        vmin = np.nanmin(val)
      # end
      if vmax < np.nanmax(val):
        vmax = np.nanmax(val)
      # end
      v_extrema = np.append(v_extrema, np.nanmin(val))
      v_extrema = np.append(v_extrema, np.nanmax(val))
    # end

    if v_extrema.size > 0:
      v_extrema = np.sort(v_extrema)
      if kwargs["cutoffglobalrange"]:
        boundary = 100 * (1 - kwargs["cutoffglobalrange"]) / 2
        vmax = np.percentile(v_extrema, 100 - boundary)
        vmin = np.percentile(v_extrema, boundary)
      # end

      if kwargs["zmin"] is None:
        kwargs["zmin"] = vmin
      # end
      if kwargs["zmax"] is None:
        kwargs["zmax"] = vmax
      # end
      if kwargs["cmin"] is None:
        kwargs["cmin"] = kwargs["zmin"]
      # end
      if kwargs["cmax"] is None:
        kwargs["cmax"] = kwargs["zmax"]
      # end
    # end
  # end

  legend_labels = None
  if kwargs.get("legend"):
    legend_labels = [label.strip() for label in kwargs["legend"].split(",")]
  # end

  kwargs["legend"] = not kwargs.get("no_legend", False)
  del kwargs["no_legend"]

  file_name = ""
  last_saved_output = None

  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    if dat.get_num_dims() != 3:
      raise click.ClickException(
          f"plot3d only supports 3D datasets. Dataset {i:d} has {dat.get_num_dims():d} dimensions."
      )
    # end

    if dataset_fignum:
      kwargs["figure"] = int(i)
    # end
    if kwargs["multiblock"]:
      kwargs["figure"] = 0
    # end

    if legend_labels is not None and i < len(legend_labels):
      label = legend_labels[i]
    elif ctx.obj["data"].get_num_datasets() > 1 or kwargs["forcelegend"]:
      label = dat.get_label()
    else:
      label = ""
    # end

    plot_kwargs = dict(kwargs)
    if slice_kwargs:
      plot_kwargs["slice_plane"] = _get_slice_kwargs_for_data(dat)
    # end

    fig = plot_output_module.plot3d(dat, label_prefix=label, **plot_kwargs)

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
          file_name = file_name + f"dataset_{i:d}"
        # end
      # end
      if kwargs["figure"] is None:
        file_name = _save_output_3d(fig, file_name)
        last_saved_output = file_name
        file_name = ""
      # end
    # end

    if kwargs["saveframes"]:
      file_name = f"{kwargs['saveframes']:s}_{i:d}.html"
      last_saved_output = _save_output_3d(fig, file_name)
      kwargs["show"] = False
    # end

    if "batch_mode" in ctx.obj and ctx.obj["batch_mode"]:
      file_name = f"{ctx.obj['saveframes_prefix']:s}_{i:d}.html"
      last_saved_output = _save_output_3d(fig, file_name)
      kwargs["show"] = False
    # end

    if not (kwargs["save"] or kwargs["saveas"]) and kwargs["show"]:
      if dat._file_name:
        preview_base = dat._file_name.split(".")[0]
      else:
        preview_base = f"plot3d_{i:d}"
      # end
      html_name = _save_output_3d(fig, base_name=preview_base, force_rotating_preview=True)
      _open_html_preview(html_name)
      kwargs["show"] = False
    # end
  # end

  if (kwargs["save"] or kwargs["saveas"]) and file_name != "":
    file_name = str(file_name)
    last_saved_output = _save_output_3d(fig, file_name)
  # end

  if kwargs["show"] and last_saved_output and os.path.exists(last_saved_output):
    _open_html_preview(last_saved_output)
  # end

  verb_print(ctx, "Finishing plot3d")
