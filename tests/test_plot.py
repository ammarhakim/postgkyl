"""Postgkyl module for testing the plotting function"""
import os
import matplotlib as mpl
import numpy as np

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