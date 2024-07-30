import os
import numpy as np

import postgkyl as pg


class TestGkyl:
  dir_path = f"{os.path.dirname(__file__)}/test_data/"

  def test_gkyl_type1(self):  # Frame without distributed memore
    data = pg.GData(f"{self.dir_path:s}shock-f-ser-p1.gkyl")
    assert np.array_equal(data.num_cells, (8, 8))

  def test_gkyl_type1_c2p(self):  # Frame with coordinate mapping
    data = pg.GData(f"{self.dir_path:s}shock-f-ser-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}shock-rtheta-ser.gkyl")
    assert np.array_equal(data.num_cells, (8, 8))

  def test_gkyl_type2(self):  # Dynvector
    data = pg.GData(f"{self.dir_path:s}twostream-field-energy.gkyl")
    assert np.array_equal(data.num_cells, (6113,))

  def test_gkyl_type3(self):  # Frame with distributed memory
    data = pg.GData(f"{self.dir_path:s}hll-euler.gkyl")
    assert np.array_equal(data.num_cells, (50, 50))

  def test_gkyl_meta(self):  # Frame with msgpack meta data included
    data = pg.GData(f"{self.dir_path:s}hll-euler.gkyl")
    assert data.ctx["frame"] == 1

  def test_gkyl_c2p_vel(self):
    data = pg.GData(f"{self.dir_path:s}bimaxwellian-elc.gkyl",
        mapc2p_vel_name=f"{self.dir_path:s}bimaxwellian-mapc2p-vel.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="gkhyb")
    dg.interpolate(overwrite=True)
    assert np.isclose(data.bounds[0][1], -1.060964e07)
    assert np.isclose(data.bounds[1][2], 1.206345e-16)

class TestAdios:
  dir_path = os.path.dirname(__file__)

  def test_adios_frame(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/twostream-f-p2.bp")
    assert np.array_equal(data.num_cells, (64, 32))

  def test_adios_dynvector(self):
    data = pg.GData(f"{self.dir_path:s}/test_data/twostream-field-energy.bp")
    assert np.array_equal(data.num_cells, (15714,))
