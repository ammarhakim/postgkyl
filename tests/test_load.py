#import pytest
import os
import numpy as np

import postgkyl as pg


class TestGkyl:
  dir_path = os.path.dirname(__file__) + '/test_data/'

  def test_gkyl_type1(self):  # Frame without distributed memore
    data = pg.GData('{:s}shock-f-ser-p1.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (8, 8))
  #end

  def test_gkyl_type1_c2p(self):  # Frame with coordinate mapping
    data = pg.GData(
      '{:s}shock-f-ser-p1.gkyl'.format(self.dir_path),
      mapc2p_name='{:s}shock-rtheta-ser.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (8, 8))
  #end

  def test_gkyl_type2(self):  # Dynvector
    data = pg.GData('{:s}firehose-field-energy.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (6400,))
  #end

  def test_gkyl_type3(self):  # Frame with distributed memory
    data = pg.GData('{:s}hll-euler.gkyl'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (50, 50))
  #end

  def test_gkyl_meta(self):  # Frame with msgpack meta data included
    data = pg.GData('{:s}hll-euler.gkyl'.format(self.dir_path))
    assert data.ctx['frame'] == 1
  #end

  def test_gkyl_c2p_vel(self):  # Frame with msgpack meta data included
    data = pg.GData('{:s}bimaxwellian-elc.gkyl'.format(self.dir_path),
                    mapc2p_vel_name='{:s}bimaxwellian-mapc2p-vel.gkyl'.format(self.dir_path))
    dg = pg.GInterpModal(data, poly_order=1, basis_type='gkhyb')
    dg.interpolate(overwrite=True)
    assert data.get_bounds()[0][1] < -1.06e+07 and data.get_bounds()[0][1] > -1.07e+07 and data.get_bounds()[1][2] > 1.2e-16 and data.get_bounds()[1][2] < 1.3e-16
  #end
#end


class TestAdios:
  dir_path = os.path.dirname(__file__)

  def test_adios_frame(self):
    data = pg.GData('{:s}/test_data/twostream-f-p2.bp'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (64, 32))
  #end

  def test_adios_dynvector(self):
    data = pg.GData('{:s}/test_data/twostream-field-energy.bp'.format(self.dir_path))
    num_cells = data.get_num_cells()
    assert np.array_equal(num_cells, (15714,))
  #end
#end