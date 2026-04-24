"""Unit tests for helper utilities in postgkyl.output."""

from __future__ import annotations

import importlib
import numpy as np

import postgkyl as pg
from postgkyl.output.axis_and_grid_prep import axis_and_grid_prep
from postgkyl.output.downsample import downsample
from postgkyl.output.load_plot_data import load_plot_data
from postgkyl.output.nodal_to_cell_centered_grid import nodal_to_cell_centered_grid


load_plot_data_module = importlib.import_module("postgkyl.output.load_plot_data")


class _FakeGData:
  def __init__(self, num_dims: int, bounds: tuple[np.ndarray, np.ndarray], cells: np.ndarray):
    self._num_dims = num_dims
    self._bounds = bounds
    self._cells = cells

  def get_num_dims(self, squeeze: bool = False) -> int:
    assert squeeze is True
    return self._num_dims

  def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
    return self._bounds

  def get_num_cells(self) -> np.ndarray:
    return self._cells


def test_downsample_any_dimension_appends_last_index():
  shape = (5, 6, 7, 8)
  a = np.arange(np.prod(shape)).reshape(shape)
  b = -a

  out_a, out_b = downsample(a, b, maximum_points_per_axis=2)

  assert out_a.shape == (3, 3, 3, 3)
  assert out_b.shape == (3, 3, 3, 3)
  np.testing.assert_array_equal(out_b, -out_a)

  expected = a[np.ix_([0, 3, 4], [0, 3, 5], [0, 4, 6], [0, 4, 7])]
  np.testing.assert_array_equal(out_a, expected)


def test_downsample_returns_input_for_bad_or_missing_limits_and_shape_mismatch():
  a = np.arange(12).reshape(3, 4)
  b = np.arange(10).reshape(2, 5)

  out = downsample(a, maximum_points_per_axis=0)
  assert out[0] is a

  out = downsample(a, maximum_points_per_axis=-3)
  assert out[0] is a

  out = downsample(a, b, maximum_points_per_axis=2)
  assert out[0] is a
  assert out[1] is b


def test_downsample_scalar_is_unchanged():
  scalar = np.array(42.0)
  out = downsample(scalar, maximum_points_per_axis=2)
  assert out[0] is scalar


def test_nodal_to_cell_centered_grid_1d_and_meshgrid_2d():
  x_nodal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  centered = nodal_to_cell_centered_grid([x_nodal], np.array([4]))
  np.testing.assert_allclose(centered[0], np.array([0.5, 1.5, 2.5, 3.5]))

  x = np.array([0.0, 1.0, 2.0, 3.0])
  y = np.array([-1.0, 0.0, 1.0])
  mx, my = nodal_to_cell_centered_grid([x, y], np.array([3, 2]), meshgrid=True)
  assert mx.shape == (3, 2)
  assert my.shape == (3, 2)
  np.testing.assert_allclose(mx[:, 0], np.array([0.5, 1.5, 2.5]))
  np.testing.assert_allclose(my[0, :], np.array([-0.5, 0.5]))


def test_nodal_to_cell_centered_grid_raises_on_dim_mismatch():
  with np.testing.assert_raises(ValueError):
    nodal_to_cell_centered_grid([np.array([0.0, 1.0, 2.0])], np.array([2, 2]))


def test_load_plot_data_tuple_mode_detects_dims_and_bounds():
  x = np.array([0.0, 1.0, 2.0, 3.0])
  y = np.array([-2.0, 0.0, 2.0])
  values = np.zeros((4, 3, 2))

  grid, out_values, num_dims, lower, upper, cells = load_plot_data(([x, y], values))

  assert num_dims == 2
  assert grid is not ([x, y])
  assert out_values is values
  np.testing.assert_allclose(lower, np.array([0.0, -2.0]))
  np.testing.assert_allclose(upper, np.array([3.0, 2.0]))
  np.testing.assert_allclose(cells, np.array([4.0, 3.0]))


def test_load_plot_data_gdata_mode_uses_gdata_metadata(monkeypatch):
  x = np.array([0.0, 1.0, 2.0, 3.0])
  y = np.array([-2.0, 0.0, 2.0])
  values = np.zeros((4, 3, 1))

  def _fake_input_parser(_):
    return [x, y], values

  monkeypatch.setattr(load_plot_data_module, "input_parser", _fake_input_parser)
  fake = _FakeGData(
      num_dims=2,
      bounds=(np.array([-1.0, -2.0]), np.array([1.0, 2.0])),
      cells=np.array([8, 9]),
  )

  _, out_values, num_dims, lower, upper, cells = load_plot_data(fake)

  assert out_values is values
  assert num_dims == 2
  np.testing.assert_array_equal(lower, np.array([-1.0, -2.0]))
  np.testing.assert_array_equal(upper, np.array([1.0, 2.0]))
  np.testing.assert_array_equal(cells, np.array([8, 9]))


def test_axis_and_grid_prep_prunes_collapsed_dims_and_formats_labels():
  x = np.linspace(0.0, 1.0, 4)
  y = np.array([0.0])
  z = np.linspace(-1.0, 1.0, 5)
  values = np.zeros((4, 1, 5, 3))

  out = axis_and_grid_prep(
      grid=[x, y, z],
      values=values,
      lower=np.array([0.0, 0.0, -1.0]),
      upper=np.array([1.0, 0.0, 1.0]),
      cells=np.array([4, 1, 5]),
      num_dims=2,
      streamline=False,
      quiver=False,
      num_axes=None,
      lineouts=None,
      xlabel=None,
      ylabel=None,
      zlabel=None,
      clabel="density",
      xshift=1.0,
      yshift=0.0,
      zshift=0.0,
      xscale=2.0,
      yscale=1.0,
      zscale=3.0,
  )

  grid, out_values, lower, upper, cells, _, num_comps, idx_comps, xlabel, ylabel, zlabel, clabel = out
  assert len(grid) == 2
  assert out_values.shape == (4, 5, 3)
  np.testing.assert_array_equal(lower, np.array([0.0, -1.0]))
  np.testing.assert_array_equal(upper, np.array([1.0, 1.0]))
  np.testing.assert_array_equal(cells, np.array([4, 5]))
  assert num_comps == 3
  assert list(idx_comps) == [0, 1, 2]
  assert xlabel == r"($z_0$ + 1.00e+00) $\times$ 2.00e+00"
  assert ylabel == r"$z_2$"
  assert zlabel == r"$z_1$ $\times$ 3.00e+00"
  assert clabel == r"density $\times$ 3.000e+00"


def test_axis_and_grid_prep_quiver_component_stride_and_lineout_xlabel():
  x = np.linspace(0.0, 1.0, 4)
  y = np.linspace(0.0, 1.0, 3)
  values = np.zeros((4, 3, 6))

  out = axis_and_grid_prep(
      grid=[x, y],
      values=values,
      lower=np.array([0.0, 0.0]),
      upper=np.array([1.0, 1.0]),
      cells=np.array([4, 3]),
      num_dims=2,
      streamline=False,
      quiver=True,
      num_axes=None,
      lineouts=1,
      xlabel=None,
      ylabel=None,
      zlabel=None,
      clabel="",
      xshift=0.0,
      yshift=0.0,
      zshift=0.0,
      xscale=1.0,
      yscale=1.0,
      zscale=1.0,
  )

  _, _, _, _, _, _, num_comps, idx_comps, xlabel, _, _, _ = out
  assert num_comps == 3
  assert list(idx_comps) == [0, 1, 2]
  assert xlabel == r"$z_1$"


def test_output_module_exports_helpers():
  assert pg.output.downsample is downsample
  assert pg.output.nodal_to_cell_centered_grid is nodal_to_cell_centered_grid
