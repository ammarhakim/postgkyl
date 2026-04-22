"""Description"""

from __future__ import annotations

import argparse
import os.path

from click import Tuple
import numpy as np
import postgkyl as pg
import pyvista as pv
from postgkyl.output.plotly import _downsample_3d_volume
from postgkyl.utils import input_parser

def _cell_centered_axis(axis_values: np.ndarray, n_cells: int) -> np.ndarray:
  """Return a cell-centered axis from nodal or centered coordinates."""
  arr = np.asarray(axis_values)
  if arr.ndim != 1:
    raise ValueError("Expected 1D coordinate axis")
  # end
  if arr.size == n_cells:
    return arr
  # end
  if arr.size == n_cells + 1:
    return 0.5 * (arr[:-1] + arr[1:])
  # end
  raise ValueError("Axis size does not match value shape")


def _centered_grid_3d(grid: list[np.ndarray], value_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return centered 3D coordinates (x, y, z) for a 3D scalar field."""
  if len(grid) < 3:
    raise ValueError("Need at least 3 grid axes for a 3D plot")
  # end
  x_axis = _cell_centered_axis(np.asarray(grid[0]), value_shape[0])
  y_axis = _cell_centered_axis(np.asarray(grid[1]), value_shape[1])
  z_axis = _cell_centered_axis(np.asarray(grid[2]), value_shape[2])
  return np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")


def pyvista(data: GData | Tuple[list, np.ndarray], args: list = (),
    show: bool = True, spin: bool = True, max_points_per_axis: int = -1, contour_levels: int = 10,
    is_log: bool = False, is_contour: bool = True, is_shaded: bool = False, hide_axes: bool = False, 
    mesh_clip_plane: bool = False, mesh_slice_plane: bool = False, volume_clip_plane: bool = False, 
    cmin: float | None = None, cmax: float | None = None, aspect_ratio: Tuple[float, float, float] = (1, 1, 1), 
    camera_azimuth: float = 0.0, camera_elevation: float = -30.0,
    opacity: str | float = 'sigmoid_4', cmap: str = 'inferno', xlabel: str = 'X', ylabel: str = "Y", zlabel: str = "Z", 
    clabel: str = "", title: str | None = "", diverging: bool = False,
    cylindrical_to_cartesian: bool = False, theme: str = "default", saveas: str = "",
    xscale: float = 1.0, yscale: float = 1.0, zscale: float = 1.0, xshift: float = 0.0, yshift: float = 0.0, zshift: float = 0.0,
    **kwargs):
  """ Description
  Creates a 3D plot of a scalar field using PyVista with various customization options.

  TODO:
  Support for animations
  """

  grid, values = input_parser(data)

  scalar = np.asarray(values[..., 0])
  x, y, z = _centered_grid_3d(grid, scalar.shape)

  if diverging:
    cmap = "RdBu_r"

  if cylindrical_to_cartesian:
    r = x
    z_cyl = y
    theta = z
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z_cyl

  # Setting the aspect ratio. (1,1,1) is a cube
  xmax, xmin = np.max(x), np.min(x)
  ymax, ymin = np.max(y), np.min(y)
  zmax, zmin = np.max(z), np.min(z)
  datamax, datamin = np.max(scalar), np.min(scalar)
  x_range = xmax - xmin
  y_range = ymax - ymin
  z_range = zmax - zmin

  # Normalize the data to fall -1 to 1, then scale by the aspect ratio. Pyvista struggles with non-integer axes limits
  x = (x - xmin) / x_range * aspect_ratio[0] * 2 - aspect_ratio[0]
  y = (y - ymin) / y_range * aspect_ratio[1] * 2 - aspect_ratio[1]
  z = (z - zmin) / z_range * aspect_ratio[2] * 2 - aspect_ratio[2]

  # Downsampling can speed up rendering
  x, y, z, scalar = _downsample_3d_volume(x,y,z,
      scalar, maximum_points_per_axis=max_points_per_axis)

  if opacity == "diverging":
    # Liner opacity. 1 on either end, 0 in the middle
    cx = np.linspace(0, 1, num=255)
    opacity = np.abs(cx - 0.5) * 2
    
  # end

  off_screen = saveas.endswith((".png", ".jpg", ".jpeg"))
  pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
  grid3d = pv.StructuredGrid(x, y, z)

  if theme != "default":
    pv.set_plot_theme(theme)
  # end

  grid3d["f_raw"] = scalar.ravel(order="F")
  data = np.asarray(grid3d["f_raw"])

  colorbarformat = "%.2e"
  if is_log:
    data = np.log10(data)
    colorbarformat = "10^%.1f"
    cmin, cmax = (np.log10(cmin) if cmin is not None else None, np.log10(cmax) if cmax is not None else None)
  # end
  grid3d["f_plot"] = data

  clim = (cmin if cmin is not None else datamin, cmax if cmax is not None else datamax) 
  scalar_bar_args = {"title": clabel, "fmt": colorbarformat}

  if is_contour:
    contours = grid3d.contour(isosurfaces=contour_levels, scalars="f_plot")
    if mesh_clip_plane:
      pl.add_mesh_clip_plane(contours, cmap=cmap, clim=clim,
        normal='-x',opacity=opacity,
        scalar_bar_args=scalar_bar_args)
    elif mesh_slice_plane:
      pl.add_mesh_slice(contours, cmap=cmap, clim=clim,
        normal='-x',opacity=opacity,
        scalar_bar_args=scalar_bar_args)
    else:
      pl.add_mesh( contours, cmap=cmap, clim=clim,
        opacity=opacity,
        scalar_bar_args=scalar_bar_args,)
  else:
    if mesh_clip_plane:
      pl.add_mesh_clip_plane(
        grid3d, scalars="f_plot", cmap=cmap, clim=clim,
        opacity=opacity,
        normal='-x',
        scalar_bar_args=scalar_bar_args,
      )
    elif mesh_slice_plane:
      pl.add_mesh_slice(
        grid3d, scalars="f_plot", cmap=cmap, clim=clim,
        opacity=opacity,
        normal='-x',
        scalar_bar_args=scalar_bar_args,
      )
    else:
      vol = pl.add_volume(
        grid3d, scalars="f_plot", cmap=cmap, clim=clim,
        opacity=opacity, shade=is_shaded,
        scalar_bar_args=scalar_bar_args,
      )
      if volume_clip_plane:
        pl.add_volume_clip_plane(
          vol, normal='-x',
        )

  if title is not None:
    pl.add_text(f"{title}", position="upper_edge", font_size=12)

  if hide_axes:
    pl.hide_axes()
  else:
    pv_bounds = pl.bounds
    bounds = (-(xmin+xshift)*xscale*pv_bounds.x_min,
              (xmax+xshift)*xscale*pv_bounds.x_max, 
              -(ymin+yshift)*yscale*pv_bounds.y_min, 
              (ymax+yshift)*yscale*pv_bounds.y_max, 
              -(zmin+zshift)*zscale*pv_bounds.z_min, 
              (zmax+zshift)*zscale*pv_bounds.z_max)
    pl.show_bounds(
      xtitle=xlabel,
      ytitle=ylabel,
      ztitle=zlabel,
      axes_ranges=bounds,
      n_xlabels=3,
      n_ylabels=3,
      n_zlabels=3,
      grid='back',
      location='origin',
      all_edges=True,
      fmt="%.2e",
    )

  # Camera rotates upon opening, breaking upon interaction
  pl.camera.azimuth = camera_azimuth
  pl.camera.elevation = camera_elevation
  if spin:
    angle = camera_azimuth
    interacting = False
    def rotate_callback(step):
        nonlocal angle, interacting
        if interacting:
            return
        angle += 0.5 
        pl.camera.azimuth = angle % 360

    def on_mouse_move(*args):
        nonlocal interacting
        interacting = True

    pl.add_timer_event(max_steps=99999999, duration=50, callback=rotate_callback)  # 20 FPS
    pl.iren.add_observer('LeftButtonPressEvent', on_mouse_move)

  if saveas != "":
    if saveas.endswith(".html"):
      pl.export_html(saveas)
    elif saveas.endswith(".pdf") or saveas.endswith(".svg"):
      pl.save_graphic(saveas)
    elif saveas.endswith(".png") or saveas.endswith(".jpg") or saveas.endswith(".jpeg"):
      pl.screenshot(saveas) #, transparent_background=True)
    elif saveas.endswith(".gltf"):
      pl.export_gltf(saveas)
    else:
      raise ValueError("Unsupported file format for saving. Supported formats are: .html, .png, .jpg, .jpeg, .pdf, .svg")

  if show:
    pl.show()
