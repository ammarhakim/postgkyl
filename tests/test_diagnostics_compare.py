"""Tests for ``postgkyl.diagnostics.compare`` -- the equation-blind
baseline-vs-candidate simulation comparison diagnostic.

Dynvector-only scenarios (ranking, selection, multiblock overlay, batch
mode) are built from real on-disk ``.gkyl`` dynvector files via
``generate_test_data.write_gkyl_dynvector``, so they exercise the actual
``pg.load``/``plot`` path without needing the compiled Gkeyll bridge.
Scenarios that also compare a spatial field (``interpolate()``) are marked
``needs_gkeyll`` and use ``write_gkyl_field``.

Run: PYTHONPATH=src pytest tests/test_diagnostics_compare.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from postgkyl import gpython
from postgkyl.diagnostics import compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "generate_test_data", os.path.join(ROOT, "tests", "generate_test_data.py"))
generate_test_data = importlib.util.module_from_spec(spec)
sys.modules.setdefault("generate_test_data", generate_test_data)
spec.loader.exec_module(generate_test_data)

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


def _write_dynvec(path, name, quantity, time, values):
  generate_test_data.write_gkyl_dynvector(
      os.path.join(path, f"{name}-{quantity}.gkyl"), np.asarray(time),
      np.asarray(values))


def _write_field(path, name, quantity, cells, values, *, poly_order=1,
                 basis_type="serendipity", frame=0):
  generate_test_data.write_gkyl_field(
      os.path.join(path, f"{name}-{quantity}_{frame}.gkyl"), cells,
      [0.0] * len(cells), [1.0] * len(cells), values, poly_order, basis_type,
      frame=frame)


def _sim(tmp_path, subdir, name="sim", *, n=5, energy_scale=1.0,
        moms_scale=1.0):
  """A minimal two-quantity dynvector-only sim: field_energy and
  elc_integrated_moms, both present under both sims by default."""
  path = str(tmp_path / subdir)
  os.makedirs(path, exist_ok=True)
  time = np.linspace(0.0, 1.0, n)
  _write_dynvec(path, name, "field_energy", time,
               (np.linspace(1.0, 2.0, n) * energy_scale)[:, None])
  moms = np.stack([
      np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n),
      np.linspace(3.0, 4.0, n)
  ], axis=-1) * moms_scale
  _write_dynvec(path, name, "elc_integrated_moms", time, moms)
  return path


def _only(reports):
  """The single :class:`CompareSimsReport` out of a non-batch ``sims()``
  call's one-entry dict."""
  assert len(reports) == 1, reports
  return next(iter(reports.values()))


class TestDefaultLabels:

  def test_shares_no_suffix_uses_basenames(self, tmp_path):
    labels = compare._default_labels(str(tmp_path / "a"), str(tmp_path / "b"))
    assert labels == ["a", "b"]

  def test_identical_basenames_use_parent_to_disambiguate(self, tmp_path):
    labels = compare._default_labels(
        str(tmp_path / "baseline" / "sim"), str(tmp_path / "candidate" / "sim"))
    assert labels == [os.path.join("baseline", "sim"),
                      os.path.join("candidate", "sim")]


class TestIsExcluded:

  def test_default_prefixes(self):
    assert compare._is_excluded("geo_int_bmag", compare._DEFAULT_EXCLUDE_PREFIXES)
    assert compare._is_excluded("bc_zlower_vcutsq",
                                compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("field_energy",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)

  def test_extra_prefix(self):
    prefixes = compare._DEFAULT_EXCLUDE_PREFIXES + ("dt",)
    assert compare._is_excluded("dt", prefixes)


class TestStemsByQuantity:

  def test_maps_quantity_to_one_stem(self, tmp_path):
    path = _sim(tmp_path, "s")
    quantities = compare._stems_by_quantity(path)
    assert quantities["field_energy"] == "sim-field_energy"
    assert quantities["elc_integrated_moms"] == "sim-elc_integrated_moms"

  def test_multiblock_stem_keeps_one_block(self, tmp_path):
    path = str(tmp_path)
    time = np.linspace(0.0, 1.0, 3)
    for block in range(3):
      _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                   np.ones((3, 1)))
    quantities = compare._stems_by_quantity(path)
    assert quantities["field_energy"] == "sim_b0-field_energy"


class TestSimsDynvectorOnly:

  def test_plots_every_shared_quantity_within_num_plots(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    assert set(report.plotted) == {"field_energy", "elc_integrated_moms"}
    assert report.only_in_0 == ()
    assert report.only_in_1 == ()
    assert report.skipped == {}
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.png"))

  def test_ranks_larger_difference_first(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", energy_scale=1.001, moms_scale=2.0)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    by_quantity = {d.quantity: d for d in report.compared}
    assert (by_quantity["elc_integrated_moms"].max_rel_diff
           > by_quantity["field_energy"].max_rel_diff)
    assert report.compared[0].quantity == "elc_integrated_moms"

  def test_num_plots_limits_auto_selection(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1))
    assert report.plotted == ("elc_integrated_moms",)

  def test_must_include_is_always_plotted(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field_energy"]))
    assert "field_energy" in report.plotted
    assert len(report.plotted) == 1

  def test_must_include_missing_from_one_sim_raises(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate")
    with pytest.raises(ValueError, match="must_include"):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["not_a_real_quantity"])

  def test_only_in_one_sim_reported_not_errored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate")
    _write_dynvec(path0, "sim", "baseline_only", np.linspace(0.0, 1.0, 3),
                 np.ones((3, 1)))
    report = _only(compare.sims(path0, path1, out_dir=str(tmp_path / "figs")))
    assert report.only_in_0 == ("baseline_only",)
    assert report.only_in_1 == ()

  def test_shape_mismatch_is_skipped_unless_must_include(self, tmp_path):
    path0 = _sim(tmp_path, "baseline", n=5)
    path1 = _sim(tmp_path, "candidate", n=4)
    report = _only(compare.sims(path0, path1, out_dir=str(tmp_path / "figs")))
    assert "field_energy" in report.skipped
    assert "field_energy" not in report.plotted

    with pytest.raises(ValueError, match="shape"):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["field_energy"])

  def test_exclude_adds_to_default_prefixes(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    exclude=["elc_"]))
    assert "elc_integrated_moms" not in [d.quantity for d in report.compared]

  def test_custom_labels_used_in_legend(self, tmp_path):
    # No assertion on rendered pixels; only that a custom label list doesn't
    # raise and is accepted end-to-end.
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    labels=["before", "after"]))
    assert report.plotted

  def test_format_controls_saved_extension(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    format="jpg"))
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.jpg"))


class TestSimsMultiblockDynvector:

  def _multiblock_sim(self, tmp_path, subdir, scale=1.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for block in range(3):
      _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                   (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    return path

  def test_overlays_every_block_onto_one_figure(self, tmp_path):
    path0 = self._multiblock_sim(tmp_path, "baseline")
    path1 = self._multiblock_sim(tmp_path, "candidate", scale=1.2)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1))
    assert report.plotted == ("field_energy",)
    assert os.path.isfile(
        os.path.join(str(tmp_path / "figs"), "field_energy.png"))
    # Exactly one PNG, not one per block.
    pngs = [f for f in os.listdir(str(tmp_path / "figs")) if f.endswith(".png")]
    assert pngs == ["field_energy.png"]


@needs_gkeyll
class TestSimsField:

  def _field_sim(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 4, 4)) + offset
    _write_field(path, "sim", "field", [4, 4], values, frame=0)
    return path

  def test_2d_field_plots_side_by_side(self, tmp_path):
    path0 = self._field_sim(tmp_path, "baseline")
    path1 = self._field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"]))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.png"))

  def _field_sim_3d(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    # serendipity p1 in 3D packs 8 basis coefficients per cell.
    values = rng.standard_normal((4, 4, 4, 8)) + offset
    _write_field(path, "sim", "field", [4, 4, 4], values, frame=0)
    return path

  def test_3d_field_defaults_to_y_midpoint(self, tmp_path):
    path0 = self._field_sim_3d(tmp_path, "baseline")
    path1 = self._field_sim_3d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"]))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.png"))

  def test_3d_field_honors_explicit_z1(self, tmp_path):
    path0 = self._field_sim_3d(tmp_path, "baseline")
    path1 = self._field_sim_3d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"], z1="0"))
    assert report.plotted == ("field",)

  def _field_sim_4d(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    # serendipity p1 in 4D packs 16 basis coefficients per cell; a 4-D
    # spatial field still has 3 dimensions left after the y-midpoint
    # default, so it needs further selectors to plot.
    values = rng.standard_normal((2, 2, 2, 2, 16)) + offset
    _write_field(path, "sim", "field", [2, 2, 2, 2], values, frame=0)
    return path

  def test_4d_field_is_unplottable_without_more_selectors(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    assert "field" not in report.plotted
    by_quantity = {d.quantity: d for d in report.compared}
    assert by_quantity["field"].plottable is False

  def test_4d_field_as_must_include_raises(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    with pytest.raises(ValueError):
      compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                  must_include=["field"])

  def test_4d_field_plots_with_explicit_selectors(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field"], z1="0", z2="0"))
    assert report.plotted == ("field",)


class TestSimsBatch:

  def test_compares_every_shared_subdirectory(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    reports = compare.sims(str(base), str(cand), batch=True,
                           out_dir=str(tmp_path / "figs"))
    assert set(reports) == {"sim_a", "sim_b"}
    for name, report in reports.items():
      assert report.plotted
      assert os.path.isdir(os.path.join(str(tmp_path / "figs"), name))

  def test_raises_when_no_subdirectory_is_shared(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    _sim(base.parent, base.name + "/sim_a", name="sim_a")
    _sim(cand.parent, cand.name + "/sim_b", name="sim_b")
    with pytest.raises(ValueError, match="no simulation subdirectory"):
      compare.sims(str(base), str(cand), batch=True,
                  out_dir=str(tmp_path / "figs"))

  def test_single_sim_dict_key_is_path0_basename(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    reports = compare.sims(path0, path1, out_dir=str(tmp_path / "figs"))
    assert set(reports) == {"baseline"}
