import os
import matplotlib as mpl

import postgkyl as pg

class TestPlot:
  """Test Postgkyl plot function.

  Currently, it doesn't properly test if the plots look correct, only if they are
  created at all (which is useful as well).
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