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
  parts = [part.strip() for part in str(value).replace(":", ",").split(",") if part.strip()]
  return (float(parts[0]), float(parts[1]))


@click.command(name="plotly-animate")
@click.option("--use", "-u", default=None, help="Tag to animate from the active dataset stack.")
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
    help="Volume and surface opacity in [0, 1].")
@click.option("--scatter-opacity-range", type=click.STRING, callback=_parse_range_option, default=None,
  help="Scatter alpha range as 'min,max' (or 'min:max'); enables opacity-gradient colorscale only when set.")
@click.option("--scatter-opacity-log/--no-scatter-opacity-log", default=False, show_default=True,
  help="Use logarithmic mapping for scatter opacity ramp.")
@click.option("--surface-count", type=click.INT, default=32, show_default=True,
    help="Number of Plotly volume isosurfaces.")
@click.option("--maximum-points-per-axis", "--mppa", "maximum_points_per_axis", type=click.INT, default=0, show_default=True,
    help="Maximum points per axis for 3D downsampling; 0 disables downsampling.")
@click.option("--background", type=click.Choice(["dark", "light"]), default="dark", show_default=True,
    help="3D scene background theme.")
@click.option("-d", "--diverging", is_flag=True, help="Use a diverging colorscale.")
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
    help="Compute a shared color range across selected datasets.")
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
@click.option("--frame-duration", type=click.INT, default=50, show_default=True,
    help="Duration of each animation frame in milliseconds.")
@click.option("--transition-duration", type=click.INT, default=0, show_default=True,
    help="Transition time between frames in milliseconds.")
@click.option("--fromcurrent/--no-fromcurrent", default=True, show_default=True,
    help="Continue animation from current frame when Play is pressed.")
@click.option("--redraw/--no-redraw", default=True, show_default=True,
    help="Force redraw on each frame.")
@click.option("--save", is_flag=True, help="Save output instead of opening preview only.")
@click.option("--saveas", type=click.STRING, default=None, help="Output HTML path for saved animation.")
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
def plotly_animate(ctx, **kwargs):
  """Animate active 2D/3D datasets with Plotly frames and playback controls."""
  verb_print(ctx, "Starting plotly-animate")
  plot_output_module = importlib.import_module("postgkyl.output.plotly")

  kwargs["rcParams"] = ctx.obj["rcParams"]

  supported_dims = (2, 3)

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
    legend_labels = [label.strip() for label in kwargs["legend"].split(",") if label.strip()]
  # end

  kwargs["legend"] = not kwargs.get("no_legend", False)
  del kwargs["no_legend"]

  frame_duration = kwargs.pop("frame_duration")
  transition_duration = kwargs.pop("transition_duration")
  fromcurrent = kwargs.pop("fromcurrent")
  redraw = kwargs.pop("redraw")

  render_kwarg_keys = {
      "squeeze", "num_axes", "num_subplot_row", "num_subplot_col",
      "scatter", "marker_radius", "markerstyle", "diverging",
      "xscale", "xshift", "yscale", "yshift", "zscale", "zshift",
      "cscale", "cshift", "cmin", "cmax", "clim",
      "background", "invert_cmap", "legend", "colorbar", "label_prefix",
      "xlabel", "ylabel", "zlabel", "clabel", "title",
      "logx", "logy", "logz", "logc", "aspect",
      "showgrid", "hashtag", "xkcd", "color", "linewidth", "opacity",
      "scatter_opacity_range", "scatter_opacity_log",
      "maximum_points_per_axis", "surface_count",
      "xrange", "yrange", "zrange", "slice_plane", "figsize",
      "cmap", "cylindrical_to_cartesian", "rcParams",
  }

  data_sequence = []
  frame_labels = []
  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    if dat.get_num_dims() not in supported_dims:
      raise click.ClickException(
          f"plotly-animate only supports 2D or 3D datasets. Dataset {i:d} has {dat.get_num_dims():d} dimensions."
      )
    # end
    data_sequence.append(dat)
    if dat.ctx.get("time") is not None:
      frame_labels.append(f"t={dat.ctx['time']:.4e}")
    elif dat.ctx.get("frame") is not None:
      frame_labels.append(f"frame {dat.ctx['frame']:d}")
    else:
      frame_labels.append(str(i))
    # end
  # end

  if not data_sequence:
    raise click.ClickException("No datasets found for plotly-animate.")
  # end

  plot_kwargs = {key: kwargs[key] for key in render_kwarg_keys if key in kwargs}

  if legend_labels is not None:
    plot_kwargs["label_prefix"] = legend_labels[0]
  elif len(data_sequence) > 1 or kwargs["forcelegend"]:
    plot_kwargs["label_prefix"] = data_sequence[0].get_label()
  else:
    plot_kwargs["label_prefix"] = ""
  # end

  fig = plot_output_module.plotly_animate(
      data_sequence,
      frame_labels=frame_labels,
      frame_duration=frame_duration,
      transition_duration=transition_duration,
      fromcurrent=fromcurrent,
      redraw=redraw,
      **plot_kwargs,
  )

  if kwargs["saveas"]:
    out_name = kwargs["saveas"]
  elif kwargs["save"]:
    out_name = "plotly-animate.html"
  else:
    out_name = os.path.join(tempfile.gettempdir(), "plotly-animate_preview.html")
  # end

  if not str(out_name).lower().endswith(".html"):
    out_name = f"{out_name}.html"
  # end

  fig.write_html(out_name)

  if kwargs["show"]:
    webbrowser.open(Path(out_name).resolve().as_uri())
  # end

  verb_print(ctx, "Finishing plotly-animate")
