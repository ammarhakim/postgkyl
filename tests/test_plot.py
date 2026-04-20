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
    fig = pg.output.plot((grid, values))
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.0, 1.0))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.0, 1.0))
    assert fig.data[0].surface.count == 32

  def test_plot_plotly_3d_ranges_override(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plot((grid, values), xrange=(0.2, 0.8), yrange=(0.1, 0.9), zrange=(0.3, 0.7), surface_count=12)
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.layout.scene.xaxis.range, (0.2, 0.8))
    np.testing.assert_allclose(fig.layout.scene.yaxis.range, (0.1, 0.9))
    np.testing.assert_allclose(fig.layout.scene.zaxis.range, (0.3, 0.7))
    assert fig.data[0].surface.count == 12

  def test_plot_plotly_3d_color_controls(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (x + y + z)[..., np.newaxis]
    fig = pg.output.plot((grid, values), cscale=2.0, cshift=1.0, clim=(1.5, 5.5))
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.data[0].cmin, 1.5)
    np.testing.assert_allclose(fig.data[0].cmax, 5.5)
    np.testing.assert_allclose(np.nanmin(fig.data[0].value), 1.0)
    np.testing.assert_allclose(np.nanmax(fig.data[0].value), 7.0)

  def test_plot_plotly_3d_logc_converts_linear_clim(self):
    grid = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]
    x, y, z = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
    values = (1.0e-2 + x + y + z)[..., np.newaxis]
    fig = pg.output.plot((grid, values), logc=True, cmin=1.0e-20, cmax=1.0e-2)
    assert isinstance(fig, go.Figure)
    np.testing.assert_allclose(fig.data[0].cmin, -20.0)
    np.testing.assert_allclose(fig.data[0].cmax, -2.0)