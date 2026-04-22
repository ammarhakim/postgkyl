import click
import numpy as np
import webbrowser

from postgkyl.utils import verb_print
import postgkyl.output.pyvista

def parse_opacity(ctx, param, value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return value
  
def parse_aspect_ratio(ctx, param, value):
  try:
    parts = value.split(',')
    if len(parts) != 3:
      raise ValueError("Aspect ratio must have three components separated by commas.")
    return tuple(float(part) for part in parts)
  except Exception as e:
    raise click.BadParameter(f"Invalid aspect ratio format: {e}")

@click.command(name="pyvista")
@click.option("--no-show", default=False, is_flag=True, help="Whether to display the plot interactively.")
@click.option("--screenshot", default=False, is_flag=True, help="Whether to save a screenshot of the plot as 'pyvista.png'.")
@click.option("--no-spin", default=False, is_flag=True, help="Whether to continuously rotate the plot for a dynamic view.")
@click.option("--max-points-per-axis", "--mppa", default=-1, type=int, help="Maximum number of points to plot along each axis (default: -1 for no downsampling).")
@click.option("--logc", default=False, is_flag=True, help="Whether to use logarithmic scaling for the color mapping.")
@click.option("--contour","-c", default=False, is_flag=True, help="Whether to display contour lines on the plot.")
@click.option("--contour-levels", default=10, type=int, help="Number of contour levels to display (default: 10).")
@click.option("--shaded", default=False, is_flag=True, help="Whether to use shaded rendering for the plot.")
@click.option("--hide-axes", default=False, is_flag=True, help="Whether to hide the axes in the plot.")
@click.option("--mesh-clip-plane", default=False, is_flag=True, help="Whether to enable clipping of the mesh with a plane.")
@click.option("--mesh-slice-plane", default=False, is_flag=True, help="Whether to enable slicing of the mesh with a plane (mutually exclusive with mesh-clip-plane).")
@click.option("--volume-clip-plane", default=False, is_flag=True, help="Whether to enable clipping of the volume with a plane.")
@click.option("--cmin", default=None, type=float, help="Minimum value for color mapping (default: data minimum).")
@click.option("--cmax", default=None, type=float, help="Maximum value for color mapping (default: data maximum).")
@click.option("--aspect-ratio", default='1,1,1', type=str, callback=parse_aspect_ratio, help="Aspect ratio for the plot as 'x,y,z' (default: '1,1,1' for equal scaling).")
@click.option("--camera-azimuth", default=0.0, type=float, help="Camera azimuth angle in degrees (default: 0.0).")
@click.option("--camera-elevation", default=-30.0, type=float, help="Camera elevation angle in degrees (default: -30.0).")
@click.option("--background", default="black", help="Background color for the plot (default: 'black').")
@click.option("--axes-color", default="white", help="Color for the axes and labels (default: 'white').")
@click.option("--opacity", default="sigmoid_4", callback=parse_opacity, help="Opacity for the volume rendering (string or float).") # pyvista also supports array inputs
@click.option("--cmap", default='inferno', help="Colormap to use for the plot (default: 'inferno').")
@click.option("--xscale", default=1.0, type=float, help="Scaling factor for the X axis (default: 1.0).")
@click.option("--yscale", default=1.0, type=float, help="Scaling factor for the Y axis (default: 1.0).")
@click.option("--zscale", default=1.0, type=float, help="Scaling factor for the Z axis (default: 1.0).")
@click.option("--xshift", default=0.0, type=float, help="Shift to apply to the X axis (default: 0.0).")
@click.option("--yshift", default=0.0, type=float, help="Shift to apply to the Y axis (default: 0.0).")
@click.option("--zshift", default=0.0, type=float, help="Shift to apply to the Z axis (default: 0.0).")
@click.option("--xlabel", default='X', help="Label for the X axis.")
@click.option("--ylabel", default='Y', help="Label for the Y axis.")
@click.option("--zlabel", default='Z', help="Label for the Z axis.")
@click.option("--clabel", default='', help="Label for the color bar (default: '').")
@click.option("--title", default='', help="Title for the plot .")
@click.option("--arg", "-a", multiple=True, help="Additional arguments to pass to the plotting function (can be specified multiple times).")
@click.option("--use", "-u", default=None, help="Specify the tag to plot.")
@click.option("--diverging", "-d", default=False, is_flag=True, help="Whether to use a diverging colormap (e.g., for data with both positive and negative values).")
@click.option("--cylindrical-to-cartesian", default=False, is_flag=True, help="Whether to convert cylindrical coordinates (r, z, theta) to Cartesian coordinates (x, y, z) for plotting.")
@click.option("--theme", default="default", help="PyVista theme to use for the plot (e.g., 'document', 'dark', 'light', etc.).")
@click.option("--saveas", default="", help="Filename to save the plot (supports .html, .pdf, .svg, png, .jpg, .jpeg).")

@click.pass_context
def pyvista(ctx, **kwargs):
  """Plot a 3D scalar field using PyVista with various customization options."""
  args = kwargs["arg"]
  # print(kwargs)
  kwargs["show"] = not kwargs["no_show"]
  kwargs["screenshot"] = kwargs["screenshot"]
  kwargs["spin"] = not kwargs["no_spin"]
  kwargs["max_points_per_axis"] = kwargs["max_points_per_axis"]
  kwargs["contour_levels"] = kwargs["contour_levels"]
  kwargs["is_log"] = kwargs["logc"]
  kwargs["is_contour"] = kwargs["contour"]
  kwargs["is_shaded"] = kwargs["shaded"]
  kwargs["hide_axes"] = kwargs["hide_axes"]
  kwargs["mesh_clip_plane"] = kwargs["mesh_clip_plane"]
  kwargs["mesh_slice_plane"] = kwargs["mesh_slice_plane"]
  kwargs["volume_clip_plane"] = kwargs["volume_clip_plane"]
  kwargs["cmin"] = kwargs["cmin"]
  kwargs["cmax"] = kwargs["cmax"]
  kwargs["aspect_ratio"] = tuple(kwargs["aspect_ratio"])
  kwargs["camera_azimuth"] = kwargs["camera_azimuth"]
  kwargs["camera_elevation"] = kwargs["camera_elevation"]
  kwargs["background"] = kwargs["background"]
  kwargs["axes_color"] = kwargs["axes_color"]
  kwargs["opacity"] = kwargs["opacity"]
  kwargs["cmap"] =kwargs["cmap"]
  kwargs["xscale"] = kwargs["xscale"]
  kwargs["yscale"] = kwargs["yscale"]
  kwargs["zscale"] = kwargs["zscale"]
  kwargs["xshift"] = kwargs["xshift"]
  kwargs["yshift"] = kwargs["yshift"]
  kwargs["zshift"] = kwargs["zshift"]
  kwargs["xlabel"] = kwargs["xlabel"]
  kwargs["ylabel"] = kwargs["ylabel"]
  kwargs["zlabel"] = kwargs["zlabel"]
  kwargs["clabel"] = kwargs["clabel"]
  kwargs["title"] = kwargs["title"]
  kwargs["diverging"] = kwargs["diverging"]
  kwargs["cylindrical_to_cartesian"] = kwargs["cylindrical_to_cartesian"]
  kwargs["theme"] = kwargs["theme"]
  kwargs["saveas"] = kwargs["saveas"]

  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    postgkyl.output.pyvista(dat, args, **kwargs)