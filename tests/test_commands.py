"""Postgkyl module for testing click commands."""
import click
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import subprocess

import postgkyl.commands as cmd
from postgkyl.pgkyl import cli

class TestCommands:
  """Base class for testing Postgkyl commands.

  Note that commands which just wrap other Postgkyl functions are not tested thoroughly,
  the goal here is to test if the command runs at all; more thorough testing should be
  delegated to the functions themselves.
  """
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  ctx = click.core.Context(cli)
  ctx.obj = {}
  ctx.obj["in_data_strings"] = [f"{dir_path:s}/twostream-f-p2.gkyl", f"{dir_path:s}/twostream-f-p2.gkyl", f"{dir_path:s}/twostream-f-p2_0.bp"]
  ctx.obj["in_data_strings_loaded"] = 0
  ctx.obj["verbose"] = False
  ctx.obj["data"] = cmd.DataSpace()

  ctx.obj["fig"] = ""
  ctx.obj["ax"] = ""

  ctx.obj["compgrid"] = None
  ctx.obj["global_var_names"] = None
  ctx.obj["global_cuts"] = (None, None, None, None, None, None, None)
  ctx.obj["global_c2p"] = None
  ctx.obj["global_c2p_vel"] = None

  ctx.obj["rcParams"] = {}

  # Check if ADIOS is isntalled
  adios_loader = importlib.util.find_spec('adios2')
  adios_missing = adios_loader is None

  # Check if ffmpeg is installed
  ffmpeg_missing = True
  try:
    subprocess.run("ffmpeg")
    ffmpeg_missing = False
  except FileNotFoundError:
    ffmpeg_missing = True
  # end

  def test_load(self):
    self.ctx.invoke(cmd.load)
    data = self.ctx.obj['data'].get_dataset(0)
    num_cells = data.num_cells
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_array_equal(num_cells, (64, 32))


  def test_ev_gkyl(self):
    # Check baseline addition
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f[0] f[0] +')
    data = self.ctx.obj['data'].get_dataset(0)
    values = data.get_values()
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_approx_equal(np.max(values), 3.352029)

    # Check longer chain, substraction, and not using dataset id
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f f + f -')
    data = self.ctx.obj['data'].get_dataset(0)
    values = data.get_values()
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_approx_equal(np.max(values), 1.676014)

    # Check tags
    self.ctx.invoke(cmd.load, tag='ts0')
    self.ctx.invoke(cmd.load, tag='ts1')
    self.ctx.invoke(cmd.ev, chain='ts0 ts0 +')
    data = self.ctx.obj['data'].get_dataset(0, tag='ts0')
    values = data.get_values()
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_approx_equal(np.max(values), 3.3520293)

    # Check ev functionality on multiple dataset together
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f[:] 2 *')
    data0 = self.ctx.obj['data'].get_dataset(0)
    values0 = data0.get_values()
    data1 = self.ctx.obj['data'].get_dataset(1)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    values1 = data1.get_values()
    np.testing.assert_approx_equal(np.max(values0), 3.3520293)
    np.testing.assert_approx_equal(np.max(values1), 3.3520293)


  @pytest.mark.skipif(adios_missing, reason="ADIOS2 is not installed")
  def test_ev_adios(self):
    # Check metadata
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f[2] f[2].charge *')
    data = self.ctx.obj['data'].get_dataset(2)
    values = data.get_values()
    charge = data.ctx["charge"]
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_approx_equal(np.min(values), -1.676014)
    # Check if metadata is properly passed through ev:
    np.testing.assert_approx_equal(charge, -1.0)


  def test_interpolate(self):
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.interpolate)
    data = self.ctx.obj['data'].get_dataset(0)
    num_cells = data.num_cells
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_array_equal(num_cells, (192, 96))


  def test_select(self):
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.select, z0='0:10', z1='0.0', comp='0,3')
    data = self.ctx.obj['data'].get_dataset(0)
    values_shape = data.values.shape
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_array_equal(values_shape, (10, 1, 2))


  def test_plot(self):
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.plot, show=False)
    fig = plt.gcf()
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    label = fig.axes[0].get_ylabel()
    plt.close("all")
    assert label == "$z_1$"


  @pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg is not installed")
  def test_animate_save(self, tmp_path):
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.load)
    fn = tmp_path / "test_anim.mp4"
    self.ctx.invoke(cmd.animate, show=False, saveas=fn)
    fig = plt.gcf()
    label = fig.get_axes()[0].get_ylabel()
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    plt.close("all")
    assert label == "$z_1$"
    assert fn.exists()


  def test_grid(self):
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.grid)
    data = self.ctx.obj['data'].get_dataset(0)
    values_shape = data.values.shape
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0
    np.testing.assert_array_equal(values_shape, (65, 33, 2))
    np.testing.assert_approx_equal(np.max(data.values[...,0]), 6.283185)
    np.testing.assert_approx_equal(np.max(data.values[...,1]), 6)