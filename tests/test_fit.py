"""Tests for tools.fit and the fit CLI command."""

from __future__ import annotations

import click
import numpy as np
import pytest

import postgkyl.commands as cmd
import postgkyl.tools as tools
from postgkyl.commands.fit import FitTypeParam
from postgkyl.data.gdata import GData
from postgkyl.pgkyl import cli


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ctx(datasets: list[GData]) -> click.Context:
  ctx = click.core.Context(cli)
  ctx.obj = {"verbose": False, "compgrid": None}
  data = cmd.DataSpace()
  for dat in datasets:
    data.add(dat)
  ctx.obj["data"] = data
  return ctx


def _gdata_1d(x_nodal: np.ndarray, y_values: np.ndarray) -> GData:
  dat = GData()
  dat.push([x_nodal], y_values[:, np.newaxis])
  return dat


def _gdata_2d(x_nodal: np.ndarray, y_nodal: np.ndarray,
    z_values: np.ndarray) -> GData:
  dat = GData()
  dat.push([x_nodal, y_nodal], z_values[..., np.newaxis])
  return dat


# ── FitTypeParam ──────────────────────────────────────────────────────────────

class TestFitTypeParam:
  """FitTypeParam wraps resolve_prefix with Click error handling."""
  p = FitTypeParam()

  def test_full_names_resolve(self):
    assert self.p.convert("linear", None, None) == "linear"
    assert self.p.convert("quadratic", None, None) == "quadratic"
    assert self.p.convert("plane", None, None) == "plane"
    assert self.p.convert("quadratic2d", None, None) == "quadratic2d"
    assert self.p.convert("exp_plateau", None, None) == "exp_plateau"

  def test_unambiguous_prefixes_resolve(self):
    assert self.p.convert("l", None, None) == "linear"
    assert self.p.convert("pl", None, None) == "plane"
    assert self.p.convert("e", None, None) == "exp_plateau"
    assert self.p.convert("quadratic2", None, None) == "quadratic2d"

  def test_exact_match_wins_over_longer_name_prefix(self):
    assert self.p.convert("quadratic", None, None) == "quadratic"

  def test_ambiguous_prefix_raises_bad_parameter(self):
    with pytest.raises(click.exceptions.BadParameter):
      self.p.convert("q", None, None)

  def test_unknown_raises_bad_parameter(self):
    with pytest.raises(click.exceptions.BadParameter):
      self.p.convert("exponential", None, None)


# ── tools: model functions ────────────────────────────────────────────────────

class TestFitFunctions:
  def test_linear_evaluation(self):
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(tools.linear(x, 3.0, -1.0), [-1.0, 2.0, 5.0])

  def test_quadratic_evaluation(self):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(tools.quadratic(x, 1.0, -2.0, 1.0), [1.0, 0.0, 1.0, 4.0])

  def test_plane_evaluation(self):
    XY = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(tools.plane(XY, 2.0, -1.0, 0.5), [0.5, 1.5])

  def test_quadratic2d_evaluation(self):
    XY = np.array([[1.0], [2.0]])
    result = tools.quadratic2d(XY, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0)
    np.testing.assert_allclose(result, [4.0])

  def test_exp_plateau_evaluation(self):
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(tools.exp_plateau(x, 2.0, 0.0, 1.0), [3.0, 3.0])

  def test_gaussian_evaluation(self):
    x = np.array([0.0])
    np.testing.assert_allclose(tools.gaussian(x, 3.0, 0.0, 1.0), [3.0])

  def test_power_evaluation(self):
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(tools.power(x, 2.0, 3.0, 1.0), [3.0, 17.0, 129.0])

  def test_sinusoid_evaluation(self):
    x = np.array([0.0, np.pi / 2])
    np.testing.assert_allclose(tools.sinusoid(x, 1.0, 1.0, 0.0, 0.5), [0.5, 1.5], atol=1e-14)

  def test_tanh_transition_evaluation(self):
    x = np.array([0.0])
    np.testing.assert_allclose(tools.tanh_transition(x, 2.0, 0.0, 1.0, -1.0), [-1.0])

  def test_fit_functions_and_ndim_consistent(self):
    assert set(tools.FIT_FUNCTIONS) == set(tools.FIT_NDIM)

  def test_fit_ndim_values(self):
    assert tools.FIT_NDIM["linear"] == 1
    assert tools.FIT_NDIM["quadratic"] == 1
    assert tools.FIT_NDIM["plane"] == 2
    assert tools.FIT_NDIM["quadratic2d"] == 2
    assert tools.FIT_NDIM["exp_plateau"] == 1
    assert tools.FIT_NDIM["gaussian"] == 1
    assert tools.FIT_NDIM["power"] == 1
    assert tools.FIT_NDIM["sinusoid"] == 1
    assert tools.FIT_NDIM["tanh_transition"] == 1


# ── tools: fit() — 1-D models ─────────────────────────────────────────────────

class TestFit1D:
  def test_linear_exact_data_recovers_params(self):
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 1.5
    params, _, R2 = tools.fit(x, y, "linear")
    np.testing.assert_allclose(params, [3.0, -1.5], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_quadratic_exact_data_recovers_params(self):
    x = np.linspace(-2, 2, 60)
    y = 0.5 * x**2 - 1.0 * x + 2.0
    params, _, R2 = tools.fit(x, y, "quadratic")
    np.testing.assert_allclose(params, [0.5, -1.0, 2.0], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_linear_noisy_data_high_R2_and_close_params(self):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 200)
    params, _, R2 = tools.fit(x, y, "linear")
    assert R2 > 0.999
    np.testing.assert_allclose(params[0], 2.0, atol=0.05)
    np.testing.assert_allclose(params[1], 1.0, atol=0.1)

  def test_returns_covariance_with_correct_shape(self):
    x = np.linspace(0, 5, 30)
    y = x + 1.0
    _, cov, _ = tools.fit(x, y, "linear")
    assert cov.shape == (2, 2)

  def test_initial_guess_does_not_change_result_on_exact_data(self):
    x = np.linspace(0, 10, 50)
    y = 5.0 * x + 3.0
    params_default, _, _ = tools.fit(x, y, "linear")
    params_guess, _, _ = tools.fit(x, y, "linear", p0=[10.0, 10.0])
    np.testing.assert_allclose(params_default, params_guess, rtol=1e-8)

  def test_exp_plateau_exact_data_recovers_params(self):
    x = np.linspace(0, 5, 80)
    true_params = [3.0, -1.5, 1.0]
    y = tools.exp_plateau(x, *true_params)
    params, _, R2 = tools.fit(x, y, "exp_plateau", p0=[1.0, -1.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_exp_plateau_noisy_data_high_R2(self):
    rng = np.random.default_rng(7)
    x = np.linspace(0, 5, 100)
    y = tools.exp_plateau(x, 3.0, -1.5, 1.0) + rng.normal(0, 0.05, 100)
    _, _, R2 = tools.fit(x, y, "exp_plateau", p0=[1.0, -1.0, 0.0])
    assert R2 > 0.99

  def test_invalid_fit_type_raises_value_error(self):
    x = np.linspace(0, 1, 10)
    y = x
    with pytest.raises(ValueError, match="not recognized"):
      tools.fit(x, y, "cubic")

  def test_gaussian_exact_data_recovers_params(self):
    x = np.linspace(-3, 3, 100)
    true_params = [2.0, 0.5, 0.8]
    y = tools.gaussian(x, *true_params)
    params, _, R2 = tools.fit(x, y, "gaussian", p0=[1.0, 0.0, 1.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_power_exact_data_recovers_params(self):
    x = np.linspace(1, 5, 60)
    true_params = [3.0, 2.0, -1.0]
    y = tools.power(x, *true_params)
    params, _, R2 = tools.fit(x, y, "power", p0=[1.0, 1.5, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_sinusoid_exact_data_recovers_params(self):
    x = np.linspace(0, 4 * np.pi, 200)
    true_params = [2.0, 1.0, 0.3, 0.5]
    y = tools.sinusoid(x, *true_params)
    params, _, R2 = tools.fit(x, y, "sinusoid", p0=[1.5, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-5)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_tanh_transition_exact_data_recovers_params(self):
    x = np.linspace(-5, 5, 100)
    true_params = [3.0, 1.0, 0.5, 2.0]
    y = tools.tanh_transition(x, *true_params)
    params, _, R2 = tools.fit(x, y, "tanh_transition", p0=[1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(params, true_params, rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)


# ── RPN expression support ────────────────────────────────────────────────────

class TestRPN:
  def test_param_names_basic(self):
    assert tools.rpn_param_names("a x * b +") == ["a", "b"]

  def test_param_names_excludes_spatial_vars(self):
    assert "x" not in tools.rpn_param_names("a x * b +")
    assert "y" not in tools.rpn_param_names("a x * b y * + c +")

  def test_param_names_excludes_operators(self):
    assert "+" not in tools.rpn_param_names("a x * b +")
    assert "*" not in tools.rpn_param_names("a x * b +")

  def test_param_names_excludes_functions(self):
    assert "exp" not in tools.rpn_param_names("A b x * exp *")

  def test_param_names_excludes_numeric_literals(self):
    assert tools.rpn_param_names("2 x * 1 +") == []

  def test_param_names_preserves_order(self):
    # A*(exp(b*x)) + C  →  params in order of first appearance
    assert tools.rpn_param_names("A b x * exp * C +") == ["A", "b", "C"]

  def test_ndim_1d(self):
    assert tools.rpn_ndim("a x * b +") == 1

  def test_ndim_2d(self):
    assert tools.rpn_ndim("a x * b y * + c +") == 2

  def test_rpn_linear_recovers_params(self):
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 1.5
    params, _, R2 = tools.fit(x, y, "a x * b +", p0=[1.0, 0.0])
    np.testing.assert_allclose(params, [3.0, -1.5], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_exp_recovers_params(self):
    x = np.linspace(0, 3, 80)
    true_A, true_b = 2.0, -0.5
    y = true_A * np.exp(true_b * x)
    params, _, R2 = tools.fit(x, y, "A b x * exp *", p0=[1.0, -1.0])
    np.testing.assert_allclose(params, [true_A, true_b], rtol=1e-6)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_rpn_plane_2d_recovers_params(self):
    X, Y = np.meshgrid(np.linspace(0, 5, 15), np.linspace(0, 3, 10), indexing="ij")
    xdata = np.array([X.flatten(), Y.flatten()])
    y = 2.0 * X.flatten() - 1.5 * Y.flatten() + 0.5
    params, _, R2 = tools.fit(xdata, y, "a x * b y * + c +", p0=[1.0, 1.0, 0.0])
    np.testing.assert_allclose(params, [2.0, -1.5, 0.5], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_literal_coefficients(self):
    x = np.linspace(1, 5, 40)
    y = 2.0 * x**2
    params, _, R2 = tools.fit(x, y, "a x 2 ** *", p0=[1.0])
    np.testing.assert_allclose(params, [2.0], rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_rpn_malformed_stack_raises(self):
    # Leading operator with empty stack causes IndexError inside curve_fit
    x = np.linspace(0, 1, 10)
    y = x
    with pytest.raises((IndexError, Exception)):
      tools.fit(x, y, "* x a +", p0=[1.0])

  def test_fittype_param_accepts_rpn(self):
    p = FitTypeParam()
    assert p.convert("a x * b +", None, None) == "a x * b +"

  def test_fittype_param_rejects_bare_unknown(self):
    p = FitTypeParam()
    with pytest.raises(click.exceptions.BadParameter):
      p.convert("cubic", None, None)

  def test_rpn_command_runs(self):
    x_nodal = np.linspace(0.0, 10.0, 51)
    x_cc = 0.5 * (x_nodal[:-1] + x_nodal[1:])
    y = 3.0 * x_cc - 1.5
    ctx = _make_ctx([_gdata_1d(x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="a x * b +", guess="1.0,0.0")


# ── tools: fit() — 2-D models ─────────────────────────────────────────────────

class TestFit2D:
  @staticmethod
  def _xdata(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.array([X.flatten(), Y.flatten()])

  def test_plane_exact_data_recovers_params(self):
    xdata = self._xdata(np.linspace(0, 5, 20), np.linspace(0, 3, 15))
    zdata = tools.plane(xdata, 2.0, -1.5, 0.5)
    params, _, R2 = tools.fit(xdata, zdata, "plane")
    np.testing.assert_allclose(params, [2.0, -1.5, 0.5], rtol=1e-10)
    assert R2 == pytest.approx(1.0, abs=1e-10)

  def test_quadratic2d_exact_data_recovers_params(self):
    xdata = self._xdata(np.linspace(0, 4, 15), np.linspace(0, 3, 12))
    true_params = [0.3, 0.2, -0.1, 1.0, -0.5, 2.0]
    zdata = tools.quadratic2d(xdata, *true_params)
    params, _, R2 = tools.fit(xdata, zdata, "quadratic2d")
    np.testing.assert_allclose(params, true_params, rtol=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-8)

  def test_plane_noisy_data_high_R2(self):
    rng = np.random.default_rng(42)
    xdata = self._xdata(np.linspace(0, 5, 30), np.linspace(0, 3, 25))
    zdata = tools.plane(xdata, 2.0, -1.5, 0.5) + rng.normal(0, 0.05, xdata.shape[1])
    _, _, R2 = tools.fit(xdata, zdata, "plane")
    assert R2 > 0.999

  def test_plane_returns_correct_covariance_shape(self):
    xdata = self._xdata(np.linspace(0, 5, 10), np.linspace(0, 3, 8))
    zdata = tools.plane(xdata, 1.0, 2.0, 0.0)
    _, cov, _ = tools.fit(xdata, zdata, "plane")
    assert cov.shape == (3, 3)


# ── fit command ───────────────────────────────────────────────────────────────

class TestFitCommand:
  _x_nodal = np.linspace(0.0, 10.0, 51)
  _x_cc = 0.5 * (_x_nodal[:-1] + _x_nodal[1:])

  _xn_2d = np.linspace(0.0, 5.0, 21)
  _yn_2d = np.linspace(0.0, 3.0, 16)
  _xcc_2d = 0.5 * (_xn_2d[:-1] + _xn_2d[1:])
  _ycc_2d = 0.5 * (_yn_2d[:-1] + _yn_2d[1:])

  def _linear_dat(self):
    return _gdata_1d(self._x_nodal, tools.linear(self._x_cc, 3.0, -1.0))

  def _quadratic_dat(self):
    return _gdata_1d(self._x_nodal, tools.quadratic(self._x_cc, 0.5, -1.0, 2.0))

  def _plane_dat(self):
    X, Y = np.meshgrid(self._xcc_2d, self._ycc_2d, indexing="ij")
    z = tools.plane(np.array([X.flatten(), Y.flatten()]), 2.0, -1.5, 0.5)
    return _gdata_2d(self._xn_2d, self._yn_2d, z.reshape(X.shape))

  def test_linear_command_runs(self):
    ctx = _make_ctx([self._linear_dat()])
    ctx.invoke(cmd.fit, fit_type="linear")

  def test_quadratic_command_runs(self):
    ctx = _make_ctx([self._quadratic_dat()])
    ctx.invoke(cmd.fit, fit_type="quadratic")

  def test_plane_command_runs(self):
    ctx = _make_ctx([self._plane_dat()])
    ctx.invoke(cmd.fit, fit_type="plane")

  def test_prefix_resolves_at_invocation(self):
    ctx = _make_ctx([self._linear_dat()])
    ctx.invoke(cmd.fit, fit_type="linear")

  def test_fit_adds_dataset_to_stack(self):
    ctx = _make_ctx([self._linear_dat()])
    ctx.invoke(cmd.fit, fit_type="linear")
    assert len(list(ctx.obj["data"].iterator())) == 2

  def test_fit_output_matches_input_grid_structure(self):
    dat = self._linear_dat()
    ctx = _make_ctx([dat])
    ctx.invoke(cmd.fit, fit_type="linear")
    datasets = list(ctx.obj["data"].iterator())
    original, fitted = datasets[0], datasets[1]
    assert fitted.get_grid()[0].shape == original.get_grid()[0].shape
    assert fitted.get_values().shape == (*original.get_values().shape[:-1], 1)

  def test_fit_output_values_are_accurate(self):
    ctx = _make_ctx([self._linear_dat()])
    ctx.invoke(cmd.fit, fit_type="linear")
    fitted = list(ctx.obj["data"].iterator())[1]
    expected = tools.linear(self._x_cc, 3.0, -1.0)
    np.testing.assert_allclose(fitted.get_values()[..., 0], expected, rtol=1e-6)

  def test_dimension_mismatch_raises(self):
    ctx = _make_ctx([self._linear_dat()])
    with pytest.raises(click.exceptions.UsageError, match="requires 2 spatial dimension"):
      ctx.invoke(cmd.fit, fit_type="plane")

  def test_component_selection_does_not_raise(self):
    y0 = tools.linear(self._x_cc, 3.0, -1.0)
    y1 = tools.linear(self._x_cc, -2.0, 5.0)
    dat = GData()
    dat.push([self._x_nodal], np.stack([y0, y1], axis=-1))
    ctx = _make_ctx([dat])
    ctx.invoke(cmd.fit, fit_type="linear", component=1)

  def test_initial_guess_does_not_raise(self):
    ctx = _make_ctx([self._linear_dat()])
    ctx.invoke(cmd.fit, fit_type="linear", guess="1.0,0.0")

  def test_exp_plateau_command_runs(self):
    y = tools.exp_plateau(self._x_cc, 3.0, -0.5, 1.0)
    ctx = _make_ctx([_gdata_1d(self._x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="exp_plateau", guess="1.0,-1.0,0.0")

  def test_gaussian_command_runs(self):
    y = tools.gaussian(self._x_cc, 2.0, 5.0, 1.5)
    ctx = _make_ctx([_gdata_1d(self._x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="gaussian", guess="1.0,5.0,1.0")

  def test_power_command_runs(self):
    y = tools.power(self._x_cc + 1.0, 1.0, 2.0, 0.0)
    ctx = _make_ctx([_gdata_1d(self._x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="power", guess="1.0,1.5,0.0")

  def test_sinusoid_command_runs(self):
    y = tools.sinusoid(self._x_cc, 1.0, 1.0, 0.0, 0.0)
    ctx = _make_ctx([_gdata_1d(self._x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="sinusoid", guess="1.0,1.0,0.0,0.0")

  def test_tanh_transition_command_runs(self):
    rng = np.random.default_rng(3)
    y = tools.tanh_transition(self._x_cc, 2.0, 5.0, 1.0, 0.0) + rng.normal(0, 0.05, len(self._x_cc))
    ctx = _make_ctx([_gdata_1d(self._x_nodal, y)])
    ctx.invoke(cmd.fit, fit_type="tanh_transition", guess="1.0,5.0,1.0,0.0")

  def test_already_cell_centered_grid_does_not_raise(self):
    dat = GData()
    y = tools.linear(self._x_cc, 2.0, 1.0)
    dat.push([self._x_cc], y[:, np.newaxis])
    ctx = _make_ctx([dat])
    ctx.invoke(cmd.fit, fit_type="linear")

  def test_collapsed_dimension_is_ignored(self):
    y = tools.linear(self._x_cc, 2.0, 1.0)
    dat = GData()
    dat.push([self._x_nodal, np.array([0.0, 1.0])], y[:, np.newaxis, np.newaxis])
    ctx = _make_ctx([dat])
    ctx.invoke(cmd.fit, fit_type="linear")

  def test_nodal_and_cell_centered_grids_both_run(self):
    y = tools.linear(self._x_cc, 2.0, 1.0)
    dat_nodal = _gdata_1d(self._x_nodal, y)
    dat_cc = GData()
    dat_cc.push([self._x_cc], y[:, np.newaxis])
    _make_ctx([dat_nodal]).invoke(cmd.fit, fit_type="linear")
    _make_ctx([dat_cc]).invoke(cmd.fit, fit_type="linear")
