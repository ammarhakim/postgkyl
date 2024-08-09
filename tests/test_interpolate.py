"""Postgkyl module for testing DG interpolation"""
import os
import numpy as np

import postgkyl as pg


class TestGkylInterpolate:
  """Test Postgkyl interpolate functions."""
  dir_path =  f"{os.path.dirname(__file__)}/test_data"

  def test_ser_p1(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="ms")
    grid, values = dg.interpolate()
    np.testing.assert_equal(len(grid[0]), 17)
    np.testing.assert_equal(len(grid[1]), 17)
    np.testing.assert_array_equal(values.shape, (16, 16, 1))
    np.testing.assert_approx_equal(values.mean(), 0.5)

  def test_ser_p2(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-f-p2.gkyl")
    dg = pg.GInterpModal(data)
    grid, values = dg.interpolate()
    np.testing.assert_equal(len(grid[0]), 193)
    np.testing.assert_equal(len(grid[1]), 97)
    np.testing.assert_array_equal(values.shape, (192, 96, 1))
    np.testing.assert_approx_equal(values.mean(), 0.08337313364405809)

  def test_ser_p1_i(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type='ms', num_interp=3)
    _, values = dg.interpolate()
    np.testing.assert_array_equal(values.shape, (24, 24, 1))
  #end

  def test_ser_p2_i(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-f-p2.gkyl")
    dg = pg.GInterpModal(data, num_interp=4)
    _, values = dg.interpolate()
    np.testing.assert_array_equal(values.shape, (256, 128, 1))
  #end

  def test_ten_p1(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ten-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="mt")
    grid, values = dg.interpolate()
    np.testing.assert_equal(len(grid[0]), 17)
    np.testing.assert_equal(len(grid[1]), 17)
    np.testing.assert_array_equal(values.shape, (16, 16, 1))
    np.testing.assert_approx_equal(values.mean(), 0.5)

  def test_ser_p1_c2p(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}/shock-rtheta-ser.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="ms")
    grid, values = dg.interpolate()
    np.testing.assert_equal(len(grid[0]), 17)
    np.testing.assert_equal(len(grid[1]), 17)
    np.testing.assert_array_equal(values.shape, (16, 16, 1))
    np.testing.assert_approx_equal(values.mean(), 0.5)

  def test_ten_p1_c2p(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ten-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}/shock-rtheta-ten.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="mt")
    grid, values = dg.interpolate()
    np.testing.assert_equal(len(grid[0]), 17)
    np.testing.assert_equal(len(grid[1]), 17)
    np.testing.assert_array_equal(values.shape, (16, 16, 1))
    np.testing.assert_approx_equal(values.mean(), 0.5)
