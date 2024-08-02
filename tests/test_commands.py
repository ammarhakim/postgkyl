import os
import matplotlib as mpl
import numpy as np
import click

import postgkyl.commands as cmd
from postgkyl.pgkyl import cli

class TestCommands:
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  ctx = click.core.Context(cli)
  ctx.obj = {}
  ctx.obj["in_data_strings"] = [f"{dir_path:s}/twostream-f-p2.bp", f"{dir_path:s}/shock-f-ser-p1.gkyl"]
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

  def test_load(self):
    self.ctx.invoke(cmd.load)
    data = self.ctx.obj['data'].get_dataset(0)
    np.testing.assert_array_equal(data.num_cells, (64, 32))
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0

  def test_ev(self):
    # Check baseline addition
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f[0] f[0] +')
    data = self.ctx.obj['data'].get_dataset(0)
    values = data.get_values()
    np.testing.assert_approx_equal(np.max(values), 2.265271)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0

    # Check longer chain, substraction, and not using dataset id
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f f + f -')
    data = self.ctx.obj['data'].get_dataset(0)
    values = data.get_values()
    np.testing.assert_approx_equal(np.max(values), 1.132636)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0

    # Check tags
    self.ctx.invoke(cmd.load, tag='ts')
    self.ctx.invoke(cmd.load, tag='shock')
    self.ctx.invoke(cmd.ev, chain='ts ts +')
    data = self.ctx.obj['data'].get_dataset(0, tag='ts')
    values = data.get_values()
    np.testing.assert_approx_equal(np.max(values), 2.265271)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0

    # Check ev functionality on multiple dataset together
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f[:] 2 *')
    data0 = self.ctx.obj['data'].get_dataset(0)
    values0 = data0.get_values()
    data1 = self.ctx.obj['data'].get_dataset(1)
    values1 = data1.get_values()
    np.testing.assert_approx_equal(np.max(values0), 2.265271)
    np.testing.assert_approx_equal(np.max(values1), 4.0)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0

    # Check metadata
    self.ctx.invoke(cmd.load)
    self.ctx.invoke(cmd.ev, chain='f f.charge *')
    data = self.ctx.obj['data'].get_dataset(0)
    values = data.get_values()
    np.testing.assert_approx_equal(np.min(values), -1.132636)
    # Check if metadata is properly passed through ev:
    np.testing.assert_approx_equal(data.ctx["charge"], -1)
    self.ctx.obj['data'].clean()
    self.ctx.obj["in_data_strings_loaded"] = 0