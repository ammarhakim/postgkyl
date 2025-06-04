"""Postgkyl module for testing data loading."""
import importlib.util
import numpy as np
import os
import pytest

import postgkyl as pg


class TestGkyl:
  """Test Gkeyll's internal binary output format."""
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  def test_gkyl_type1(self):  # Frame without distributed memory
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    np.testing.assert_array_equal(data.num_cells, (8, 8))

  def test_gkyl_type1_partial(self):  # Partial frame without distributed memory
    data = pg.GData(f"{self.dir_path:s}/twostream-f-p2.gkyl",
        z0='16', z1='8:-8', comp='0')
    np.testing.assert_array_equal(data.values.shape, (1, 16, 1))

  def test_gkyl_type1_c2p(self):  # Frame with coordinate mapping
    data = pg.GData(f"{self.dir_path:s}/shock-f-ser-p1.gkyl",
        mapc2p_name=f"{self.dir_path:s}/shock-rtheta-ser.gkyl")
    np.testing.assert_array_equal(data.num_cells, (8, 8))

  def test_gkyl_type2(self):  # Dynvector
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.gkyl")
    np.testing.assert_array_equal(data.num_cells, (6113,))

  def test_gkyl_type3(self):  # Frame with distributed memory
    data = pg.GData(f"{self.dir_path:s}/hll-euler.gkyl")
    np.testing.assert_array_equal(data.num_cells, (50, 50))

  def test_gkyl_type3_partial(self):  # Partial frame with distributed memory
    data = pg.GData(f"{self.dir_path:s}/hll-euler.gkyl",
        z0=30, z1='30:-5', comp=0)
    np.testing.assert_array_equal(data.values.shape, (1, 15, 1))

  def test_gkyl_meta(self):  # Frame with msgpack meta data included
    data = pg.GData(f"{self.dir_path:s}/hll-euler.gkyl")
    np.testing.assert_equal(data.ctx["frame"], 1)

  def test_gkyl_c2p_vel(self):
    data = pg.GData(f"{self.dir_path:s}/bimaxwellian-elc.gkyl",
        mapc2p_vel_name=f"{self.dir_path:s}/bimaxwellian-mapc2p-vel.gkyl")
    dg = pg.GInterpModal(data, poly_order=1, basis_type="gkhyb")
    dg.interpolate(overwrite=True)
    np.testing.assert_approx_equal(data.bounds[0][1], -1.060964e07)
    np.testing.assert_approx_equal(data.bounds[1][2], 1.206345e-16)
class TestAdios:
  """Test Gkeyll's ADIOS2 output format."""
  dir_path =  f"{os.path.dirname(__file__)}/test_data"

  adios_loader = importlib.util.find_spec('adios2')
  adios_missing = adios_loader is None

  @pytest.mark.skipif(adios_missing, reason="ADIOS2 is not installed")
  def test_adios_frame(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-f-p2_0.bp")
    np.testing.assert_array_equal(data.num_cells, (64, 32))

  @pytest.mark.skipif(adios_missing, reason="ADIOS2 is not installed")
  def test_adios_frame_partial(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-f-p2_0.bp", z0=32, comp=0)
    np.testing.assert_array_equal(data.values.shape, (1, 32, 1))

  @pytest.mark.skipif(adios_missing, reason="ADIOS2 is not installed")
  def test_adios_dynvector(self):
    data = pg.GData(f"{self.dir_path:s}/twostream-field-energy.bp")
    np.testing.assert_array_equal(data.num_cells, (15714,))
