#import pytest
import os
import numpy as np

import postgkyl as pg


class TestGkyl:
  dir_path = os.path.dirname(__file__)

  def test_gkyl_type1(self):
    data = pg.GData('{:s}/test_data/shock-f-ser-p1.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (8, 8))
  #end

  def test_gkyl_type1_c2p(self):
    data = pg.GData(
      '{:s}/test_data/shock-f-ser-p1.gkyl'.format(self.dir_path),
      mapc2p_name='{:s}/test_data/shock-rtheta-ser.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (8, 8))
  #end

  def test_gkyl_type2(self):
    data = pg.GData('{:s}/test_data/firehose-field-energy.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (6400,))
  #end

  def test_gkyl_type3(self):
    data = pg.GData('{:s}/test_data/hll-euler.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (50, 50))
  #end

  def test_gkyl_meta(self):
    data = pg.GData('{:s}/test_data/hll-euler.gkyl'.format(self.dir_path))
    assert data.ctx['frame'] == 1
  #end
#end


class TestAdios:
  dir_path = os.path.dirname(__file__)

#end