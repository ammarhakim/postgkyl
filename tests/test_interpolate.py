import os
import numpy as np

import postgkyl as pg


class TestGkylInterpolate:
  dir_path = os.path.dirname(__file__)

  def test_ser_p1(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/shock-f-ser-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="ms")
    grid, values = dg.interpolate()
    assert len(grid[0]) == 17
    assert len(grid[1]) == 17
    assert np.array_equal(values.shape, (16, 16, 1))
    assert np.isclose(values.mean(), 0.5)

  def test_ser_p2(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/twostream-f-p2.gkyl")
    dg = pg.GInterpModal(data)
    grid, values = dg.interpolate()
    assert len(grid[0]) == 193
    assert len(grid[1]) == 97
    assert np.array_equal(values.shape, (192, 96, 1))
    assert np.isclose(values.mean(), 0.08337313364405809)

  def test_ser_p1_i(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/shock-f-ser-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type='ms', num_interp=3)
    grid, values = dg.interpolate()
    assert np.array_equal(values.shape, (24, 24, 1))
  #end

  def test_ser_p2_i(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/twostream-f-p2.gkyl")
    dg = pg.GInterpModal(data, num_interp=4)
    grid, values = dg.interpolate()
    assert np.array_equal(values.shape, (256, 128, 1))
  #end

  def test_ten_p1(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/shock-f-ten-p1.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="mt")
    grid, values = dg.interpolate()
    assert len(grid[0]) == 17
    assert len(grid[1]) == 17
    assert np.array_equal(values.shape, (16, 16, 1))
    assert np.isclose(values.mean(), 0.5)

  def test_ser_p1_c2p(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/shock-f-ser-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}/test_data/shock-rtheta-ser.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="ms")
    grid, values = dg.interpolate()
    assert len(grid[0]) == 17
    assert len(grid[1]) == 17
    assert np.array_equal(values.shape, (16, 16, 1))
    assert np.isclose(values.mean(), 0.5)

  def test_ten_p1_c2p(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/shock-f-ten-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}/test_data/shock-rtheta-ten.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="mt")
    grid, values = dg.interpolate()
    assert len(grid[0]) == 17
    assert len(grid[1]) == 17
    assert np.array_equal(values.shape, (16, 16, 1))
    assert np.isclose(values.mean(), 0.5)
