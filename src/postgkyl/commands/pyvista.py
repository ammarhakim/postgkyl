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
@click.option("--no-contour","-c", default=False, is_flag=True, help="Enables full volume rendering (expensive).")
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
@click.option("--opacity", "-o", default="sigmoid_4", callback=parse_opacity, help="Opacity for the volume rendering (string or float).") # pyvista also supports array inputs
@click.option("--cmap", default='inferno', help="Colormap to use for the plot (default: 'inferno').")
@click.option("--xscale", default=1.0, type=float, help="Scaling factor for the X axis (default: 1.0).")
@click.option("--yscale", default=1.0, type=float, help="Scaling factor for the Y axis (default: 1.0).")
@click.option("--zscale", default=1.0, type=float, help="Scaling factor for the Z axis (default: 1.0).")
@click.option("--xshift", default=0.0, type=float, help="Shift to apply to the X axis (default: 0.0).")
@click.option("--yshift", default=0.0, type=float, help="Shift to apply to the Y axis (default: 0.0).")
@click.option("--zshift", default=0.0, type=float, help="Shift to apply to the Z axis (default: 0.0).")
@click.option("--xlabel", default=None, help="Label for the X axis (default: inferred, e.g. '$z_0$').")
@click.option("--ylabel", default=None, help="Label for the Y axis (default: inferred, e.g. '$z_1$').")
@click.option("--zlabel", default=None, help="Label for the Z axis (default: inferred, e.g. '$z_2$').")
@click.option("--clabel", default='', help="Label for the color bar (default: '').")
@click.option("--title", default='', help="Title for the plot .")
@click.option("--arg", "-a", multiple=True, help="Additional arguments to pass to the plotting function (can be specified multiple times).")
@click.option("--use", "-u", default=None, help="Specify the tag to plot.")
@click.option("--diverging", "-d", default=False, is_flag=True, help="Whether to use a diverging colormap (e.g., for data with both positive and negative values).")
@click.option("--cylindrical-to-cartesian", default=False, is_flag=True, help="Whether to convert cylindrical coordinates (r, z, theta) to Cartesian coordinates (x, y, z) for plotting.")
@click.option("--theme", default="default", help="PyVista theme to use for the plot (e.g., 'document', 'dark', 'light', etc.).")
@click.option("--saveas", default="", help="Filename to save the plot (supports .html, .pdf, .svg, png, .jpg, .jpeg, .gltf).")
@click.option("--hide-zeros", default=False, is_flag=True, help="Whether to hide zero values in the plot.")

@click.pass_context
def pyvista(ctx, **kwargs):
  """Plot a 3D scalar field using PyVista with various customization options."""
  args = kwargs["arg"]
  kwargs.update(
    show=not kwargs["no_show"],
    spin=not kwargs["no_spin"],
    is_log=kwargs["logc"],
    is_contour=not kwargs["no_contour"],
    is_shaded=kwargs["shaded"],
    aspect_ratio=tuple(kwargs["aspect_ratio"]),
    cylindrical_to_cartesian=kwargs["cylindrical_to_cartesian"],
  )
  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    postgkyl.output.pyvista(dat, args, **kwargs)