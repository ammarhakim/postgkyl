"""Postgkyl module for testing the plotting function"""
import os
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

import postgkyl as pg

class TestPlot:
  """Test Postgkyl plot function.

  Currently, this tests if plots look OK only to some extend (by checking plotted
  values) and mostly tests if plots are created at all. Testing images themselves is
  complicated and differs based on system and/or backend used.
  """
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  def test_plot_pcolormesh(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data)
    assert isinstance(img, mpl.collections.QuadMesh)
    mpl.pyplot.close("all")

  def test_plot_contour(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data, contour=True)
    assert isinstance(img, mpl.contour.QuadContourSet)
    mpl.pyplot.close("all")

  def test_plot_contour_options(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data, contour=True, cnlevels=5, cont_label=True)
    assert isinstance(img, mpl.contour.QuadContourSet)
    mpl.pyplot.close("all")

  def test_plot_line(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.gkyl")
    img = pg.output.plot(data)
    assert isinstance(img[0], mpl.lines.Line2D)
    mpl.pyplot.close("all")

    pg.data.select(data, comp=0, overwrite=True)
    img = pg.output.plot(data)
    x_plot, y_plot = img[0].get_xydata().T
    np.testing.assert_array_almost_equal(data.get_grid()[0], x_plot)
    np.testing.assert_array_almost_equal(data.get_values()[...,0], y_plot)
    mpl.pyplot.close("all")

  def test_plot_plotly_3d(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values))
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.0, 1.0))
    assert fig.data[0].surface.count == 32

  def test_plot_plotly_2d_surface(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 5)]
    x, y = np.meshgrid(grid[0], grid[1], indexing="ij")
    values = (x + 2.0 * y)[..., np.newaxis]
    fig = pg.output.plotly((grid, values))
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)
    np.testing.assert_allclose(fig.data[0].z, x + 2.0 * y)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.0, 3.0))

  def test_plot_plotly_2d_surface_animation(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 5)]
    x, y = np.meshgrid(grid[0], grid[1], indexing="ij")
    values0 = (x + 2.0 * y)[..., np.newaxis]
    values1 = (x + 2.0 * y + 0.5)[..., np.newaxis]
    fig = pg.output.plotly_animate([(grid, values0), (grid, values1)], frame_duration=40)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)
    assert len(fig.frames) == 1
    assert fig.frames[0].name == "1"
    assert fig.layout.updatemenus[0].buttons[0].label == "Play"

  def test_plot_plotly_3d_ranges_override(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), xrange=(0.2, 0.8), yrange=(0.1, 0.9), zrange=(0.3, 0.7), surface_count=12)
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.2, 0.8))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.1, 0.9))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.3, 0.7))
    assert fig.data[0].surface.count == 12

  def test_plot_plotly_3d_color_controls(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), cscale=2.0, cshift=1.0, clim=(1.5, 5.5))
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.data[0].cmin, 1.5)
    np.testing.assert_allclose(fig.data[0].cmax, 5.5)
    np.testing.assert_allclose(np.nanmin(fig.data[0].value), 1.0)
    np.testing.assert_allclose(np.nanmax(fig.data[0].value), 7.0)

  def test_plot_plotly_3d_logc_converts_linear_clim(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (1.0e-2 + x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), logc=True, cmin=1.0e-20, cmax=1.0e-2)
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.data[0].cmin, -20.0)
    np.testing.assert_allclose(fig.data[0].cmax, -2.0)

  def test_plot_plotly_3d_fix_aspect_uses_cube_mode(self):
    grid = [np.linspace(0.0, 2.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), aspect="cube")
    assert isinstance(fig, go.Figure)
    assert fig.layout.scene.aspectmode == "cube"

  def test_plot_plotly_3d_aspect_string_sets_mode(self):
    grid = [np.linspace(0.0, 2.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), aspect="data")
    assert isinstance(fig, go.Figure)
    assert fig.layout.scene.aspectmode == "data"

  def test_plot_plotly_3d_aspect_numeric_sets_manual_ratio(self):
    grid = [np.linspace(0.0, 2.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 0.5, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), aspect=2.0)
    assert isinstance(fig, go.Figure)
    assert fig.layout.scene.aspectmode == "manual"
    assert fig.layout.scene.aspectratio.x == 2.0
    assert fig.layout.scene.aspectratio.y == 2.0
    assert fig.layout.scene.aspectratio.z == 2.0

  def test_plot_plotly_3d_cylindrical_to_cartesian(self):
    r = np.linspace(0.0, 1.0, 4)
    z = np.linspace(-0.5, 0.5, 4)
    phi = np.linspace(0.0, 2.0 * np.pi, 5)
    rr, zz, pp = np.meshgrid(r, z, phi, indexing="ij")
    values = (rr + zz)[..., np.newaxis]
    fig = pg.output.plotly(([r,z,phi], values), cylindrical_to_cartesian=True)
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (-1.0, 1.0), atol=1.0e-12)
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (-1.0, 1.0), atol=1.0e-12)
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (-0.5, 0.5), atol=1.0e-12)

  def test_plot_plotly_3d_scatter_trace(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), scatter=True, marker_radius=3.0, markerstyle="square", cmin=0.2, cmax=2.8)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Scatter3d)
    assert fig.data[0].mode == "markers"
    np.testing.assert_allclose(fig.data[0].marker.size, 6.0)
    assert fig.data[0].marker.symbol == "square"
    np.testing.assert_allclose(fig.data[0].marker.cmin, 0.2)
    np.testing.assert_allclose(fig.data[0].marker.cmax, 2.8)

  def test_plot_plotly_3d_scatter_downsampling(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), scatter=True, maximum_points_per_axis=2)
    assert isinstance(fig, go.Figure)
    # For each axis: size 4 downsampled to indices [0, 2, 3] => 3 points per axis.
    assert len(fig.data[0].x) == 27
    assert len(fig.data[0].y) == 27
    assert len(fig.data[0].z) == 27

  def test_plot_plotly_3d_scatter_uses_opacity_gradient_when_requested(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), scatter=True, opacity=0.5, scatter_opacity_range=(0.01, 1.0))
    assert isinstance(fig, go.Figure)
    colorscale = fig.data[0].marker.colorscale
    low_color = colorscale[0][1]
    high_color = colorscale[-1][1]
    low_alpha = float(low_color.split(",")[-1].rstrip(")"))
    high_alpha = float(high_color.split(",")[-1].rstrip(")"))
    assert low_alpha < high_alpha

  def test_plot_plotly_3d_scatter_keeps_uniform_opacity_by_default(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly((grid, values), scatter=True, opacity=0.5)
    assert isinstance(fig, go.Figure)
    colorscale = fig.data[0].marker.colorscale
    low_color = colorscale[0][1]
    high_color = colorscale[-1][1]
    low_alpha = float(low_color.split(",")[-1].rstrip(")"))
    high_alpha = float(high_color.split(",")[-1].rstrip(")"))
    np.testing.assert_allclose(low_alpha, high_alpha)
    np.testing.assert_allclose(fig.data[0].marker.opacity, 0.5)

  def test_plot_plotly_3d_scatter_uses_log_opacity_ramp_when_requested(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plotly(
        (grid, values),
        scatter=True,
        scatter_opacity_range=(0.01, 1.0),
        scatter_opacity_log=True,
    )
    assert isinstance(fig, go.Figure)
    colorscale = fig.data[0].marker.colorscale

    alphas = np.array([
        float(color.split(",")[-1].rstrip(")"))
        for _, color in colorscale
    ])

    q1 = int(0.25 * (len(alphas) - 1))
    q2 = int(0.50 * (len(alphas) - 1))
    q3 = int(0.75 * (len(alphas) - 1))
    low_span = alphas[q1] - alphas[0]
    high_span = alphas[-1] - alphas[q3]
    assert low_span > high_span