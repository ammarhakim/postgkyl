import click
import importlib
import numpy as np
import os.path
from pathlib import Path
import tempfile
import webbrowser

from postgkyl.utils import verb_print


def _parse_range_option(_ctx, _param, value):
  if value is None:
    return None
  # end
  # Convert "lower,upper" or "lower:upper" into a tuple of floats (lower, upper)
  parts = [part.strip() for part in str(value).replace(":", ",").split(",") if part.strip()]
  return (float(parts[0]), float(parts[1]))

def _parse_slice_option(_ctx, _param, value):
  if value is None:
    return None
  # end
  tokens = [token.strip() for token in str(value).split(",") if token.strip()]
  selectors = []
  for token in tokens:
    token_lower = token.lower()
    if "." in token_lower or "e" in token_lower:
      selectors.append(float(token))
    else:
      selectors.append(int(token))
    # end
  # end
  return selectors


@click.command(name="plot3d")
@click.option("--use", "-u", default=None, help="Tag to plot from the active dataset stack.")
@click.option("--squeeze", is_flag=True, help="Draw all components in a single 3D scene.")
@click.option("--subplots", "-b", is_flag=True, help="Draw components in separate 3D subplots.")
@click.option("--nsubplotrow", "num_subplot_row", type=click.INT,
    help="Number of subplot rows for multi-component 3D plots.")
@click.option("--nsubplotcol", "num_subplot_col", type=click.INT,
    help="Number of subplot columns for multi-component 3D plots.")
@click.option("-s", "--scatter", is_flag=True,
  help="Render point samples as sphere-like colored markers.")
@click.option("--marker-radius", type=click.FLOAT, default=4.0, show_default=True,
  help="Scatter marker radius in pixels.")
@click.option("--markerstyle", type=click.Choice([
    "circle", "square", "diamond", "cross", "x",
]), default="circle", show_default=True,
  help="Marker shape for scatter points.")
@click.option("-o", "--opacity", type=click.FLOAT, default=1.0, show_default=True,
    help="Volume and slice opacity in [0, 1].")
@click.option("--scatter-opacity-range", type=click.STRING, callback=_parse_range_option, default=None,
  help="Scatter alpha range as 'min,max' (or 'min:max'); enables opacity-gradient colorscale only when set.")
@click.option("--scatter-opacity-log/--no-scatter-opacity-log", default=False, show_default=True,
  help="Use logarithmic mapping for scatter opacity ramp (rapid low-end change, flatter high-end).")
@click.option("--surface-count", type=click.INT, default=32, show_default=True,
    help="Number of Plotly volume isosurfaces.")
@click.option("--maximum-points-per-axis", "--mppa", "maximum_points_per_axis", type=click.INT, default=0, show_default=True,
    help="Maximum points per axis for 3D downsampling; 0 disables downsampling.")
@click.option("--background", type=click.Choice(["dark", "light"]), default="dark", show_default=True,
    help="3D scene background theme.")
@click.option("-d", "--diverging", is_flag=True, help="Use a diverging colorscale.")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Use equal scaling on x/y/z axes.")
@click.option("--aspect", default=None,
    help="Aspect mode: auto, data, cube, or a numeric uniform ratio.")
@click.option("--logx", is_flag=True, help="Use log scaling on x axis.")
@click.option("--logy", is_flag=True, help="Use log scaling on y axis.")
@click.option("--logz", is_flag=True, help="Use log scaling on z axis.")
@click.option("--logc", is_flag=True, help="Use log scaling for scalar coloring.")
@click.option("--xshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Additive shift for x coordinates.")
@click.option("--yshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Additive shift for y coordinates.")
@click.option("--zshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Additive shift for scalar values before coloring.")
@click.option("--cshift", default=0.0, type=click.FLOAT, show_default=True,
    help="Additive shift for color-mapped values.")
@click.option("--xscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Multiplicative scale for x coordinates.")
@click.option("--yscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Multiplicative scale for y coordinates.")
@click.option("--zscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Multiplicative scale for scalar values before coloring.")
@click.option("--cscale", default=1.0, type=click.FLOAT, show_default=True,
    help="Multiplicative scale for color-mapped values.")
@click.option("--slice-at-z0", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Slice selectors along z0: comma-separated, ints=index, floats=coordinate.")
@click.option("--slice-at-z1", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Slice selectors along z1: comma-separated, ints=index, floats=coordinate.")
@click.option("--slice-at-z2", type=click.STRING, callback=_parse_slice_option, default=None,
    help="Slice selectors along z2: comma-separated, ints=index, floats=coordinate.")
@click.option("--xlim", default=None, type=click.STRING, callback=_parse_range_option,
    help="x-axis limits as 'lower,upper' (or 'lower:upper').")
@click.option("--ylim", default=None, type=click.STRING, callback=_parse_range_option,
    help="y-axis limits as 'lower,upper' (or 'lower:upper').")
@click.option("--zlim", default=None, type=click.STRING, callback=_parse_range_option,
    help="z-axis limits as 'lower,upper' (or 'lower:upper').")
@click.option("--clim", default=None, type=click.STRING, callback=_parse_range_option,
    help="Color limits as 'lower,upper' (or 'lower:upper').")
@click.option("--cmax", default=None, type=click.FLOAT, help="Maximum color value.")
@click.option("--cmin", default=None, type=click.FLOAT, help="Minimum color value.")
@click.option("--globalrange", "-r", is_flag=True,
    help="Compute a shared color range across selected 3D datasets.")
@click.option("--cutoffglobalrange", "-cogr", default=None, type=click.FLOAT,
    help="Percentile cutoff for shared color range (e.g. 0.98).")
@click.option("--legend", default=None, type=click.STRING,
    help="Comma-separated legend labels for datasets.")
@click.option("--no-legend", is_flag=True, help="Hide legend labels.")
@click.option("--force-legend", "forcelegend", is_flag=True,
    help="Force legend labels even for single dataset plots.")
@click.option("--color", type=click.STRING, help="Use a fixed color (bypasses colorscale).")
@click.option("-x", "--xlabel", type=click.STRING, help="x-axis label.")
@click.option("-y", "--ylabel", type=click.STRING, help="y-axis label.")
@click.option("-z", "--zlabel", type=click.STRING, help="z-axis label.")
@click.option("--clabel", type=click.STRING, help="Colorbar label.")
@click.option("--title", type=click.STRING, help="Figure title.")
@click.option("--save", is_flag=True, help="Save output instead of opening preview only.")
@click.option("--saveas", type=click.STRING, default=None, help="Output path for saved figure.")
@click.option("--starting-azimuthal-angle", "azimuthal_angle", "--azimuthal-angle",
    type=click.FLOAT, default=0.0, show_default=True,
    help="Starting azimuthal camera angle in degrees for rotating exports.")
@click.option("--polar-angle", type=click.FLOAT, default=85.0, show_default=True,
    help="Polar camera angle in degrees for rotating exports.")
@click.option("--rotation-period", type=click.FLOAT, default=20.0, show_default=True,
    help="Seconds per full camera rotation for rotating exports.")
@click.option("--fps", type=click.INT, default=1, show_default=True,
    help="Frames-per-second for rotating mp4/gif output.")
@click.option("--showgrid/--no-showgrid", default=True, help="Show 3D axis grid planes.")
@click.option("--hashtag", is_flag=True, help="Add '#pgkyl' annotation to the figure.")
@click.option("--show/--no-show", default=True,
    help="Open the output preview in a browser.")
@click.option("--figsize", help="Figure size as 'width,height' (scaled to pixels for Plotly).")
@click.option("--cmap", type=click.STRING, default=None,
    help="Set a matplotlib colormap name for Plotly colorscale conversion.")
@click.option("--invert-cmap", is_flag=True,
    help="Invert the chosen colormap.")
@click.option("--cylindrical-to-cartesian", is_flag=True,
  help="Interpret (z0, z1, z2) as (R, Z, phi) and convert to Cartesian (x, y, z).")
@click.pass_context
def plot3d(ctx, **kwargs):
  """Plot active 3D datasets, or 2D datasets as 3D surfaces, with Plotly."""
  verb_print(ctx, "Starting plot3d")
  plot_output_module = importlib.import_module("postgkyl.output.plot3d")

  def _save_output_3d(fig, file_name: str | None = None, base_name: str | None = None,
      force_rotating_preview: bool = False) -> str:
    if force_rotating_preview:
      safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (base_name or "")).strip("_")
      if not safe_base:
        safe_base = "plot3d_preview"
      # end
      file_name = os.path.join(tempfile.gettempdir(), f"{safe_base}_preview.html")
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

  if kwargs["aspect"]:
    kwargs["fixaspect"] = True
  # end

  supported_dims = (2, 3)

  slice_kwargs = {}
  for d in range(3):
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
    for dat in ctx.obj["data"].iterator(kwargs["use"]):
      kwargs["num_axes"] = kwargs["num_axes"] + dat.get_num_comps()
    # end
  # end

  if kwargs["xlim"]:
    kwargs["xrange"] = kwargs["xlim"]
  # end
  if kwargs["ylim"]:
    kwargs["yrange"] = kwargs["ylim"]
  # end
  if kwargs["zlim"]:
    kwargs["zrange"] = kwargs["zlim"]
  # end
  if kwargs["clim"]:
    kwargs["cmin"], kwargs["cmax"] = kwargs["clim"]
  # end

  if kwargs["globalrange"] or kwargs["cutoffglobalrange"]:
    vmin = float("inf")
    vmax = float("-inf")
    v_extrema = np.array([])
    for dat in ctx.obj["data"].iterator(kwargs["use"]):
      if dat.get_num_dims() not in supported_dims:
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

      if kwargs["cmin"] is None:
        kwargs["cmin"] = vmin
      # end
      if kwargs["cmax"] is None:
        kwargs["cmax"] = vmax
      # end
    # end
  # end

  legend_labels = None
  if kwargs.get("legend"):
    legend_labels = [label.strip() for label in kwargs["legend"].split(",")]
  # end

  kwargs["legend"] = not kwargs.get("no_legend", False)
  del kwargs["no_legend"]

  render_kwarg_keys = {
      "squeeze", "num_axes", "num_subplot_row", "num_subplot_col",
      "scatter", "marker_radius", "markerstyle", "diverging",
      "xscale", "xshift", "yscale", "yshift", "zscale", "zshift",
      "cscale", "cshift", "cmin", "cmax", "clim",
      "background", "invert_cmap", "legend", "colorbar", "label_prefix",
      "xlabel", "ylabel", "zlabel", "clabel", "title",
      "logx", "logy", "logz", "logc", "fixaspect", "aspect",
      "showgrid", "hashtag", "xkcd", "color", "linewidth", "opacity",
      "scatter_opacity_range", "scatter_opacity_log",
      "maximum_points_per_axis", "surface_count",
      "xrange", "yrange", "zrange", "slice_plane", "figsize",
        "cmap", "cylindrical_to_cartesian", "rcParams",
  }

  file_name = ""
  last_saved_output = None

  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    if dat.get_num_dims() not in supported_dims:
      raise click.ClickException(
          f"plot3d only supports 2D or 3D datasets. Dataset {i:d} has {dat.get_num_dims():d} dimensions."
      )
    # end

    if legend_labels is not None and i < len(legend_labels):
      label = legend_labels[i]
    elif ctx.obj["data"].get_num_datasets() > 1 or kwargs["forcelegend"]:
      label = dat.get_label()
    else:
      label = ""
    # end

    plot_kwargs = {key: kwargs[key] for key in render_kwarg_keys if key in kwargs}
    if slice_kwargs:
      plot_kwargs["slice_plane"] = _get_slice_kwargs_for_data(dat)
    # end
    plot_kwargs["label_prefix"] = label

    fig = plot_output_module.plot3d(dat, **plot_kwargs)

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
      last_saved_output = _save_output_3d(fig, file_name)
      file_name = ""
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

  if kwargs["show"] and last_saved_output and os.path.exists(last_saved_output):
    _open_html_preview(last_saved_output)
  # end

  verb_print(ctx, "Finishing plot3d")
