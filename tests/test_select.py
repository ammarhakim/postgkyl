"""Postgkyl module for testing data loading."""
import numpy as np
import os

import postgkyl as pg


class TestSelect:
  """Test Gkeyll's select commands."""
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  def test_integer(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    grid, data = pg.data.select(data, z0=1, z1="5:8")
    np.testing.assert_array_equal(grid[0], [1.375, 1.75 ])
    np.testing.assert_array_almost_equal(grid[1], [3.926991, 4.712389, 5.497787, 6.283185])
    np.testing.assert_array_equal(data.shape, (1, 3, 4))

  def test_float(self):
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    grid, data = pg.data.select(data, z0=0.5)
    np.testing.assert_array_equal(grid[0], [1.0, 1.375])
