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
import matplotlib.pyplot as plt
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


def _pdf_page_count(path) -> int:
  """Count pages in a PDF by scanning its raw bytes for ``/Type /Page``
  (excluding ``/Type /Pages``) -- neither ``pypdf`` nor ``pdfminer`` is
  installed in this environment, but this is enough to verify page counts
  against a real ``matplotlib.backends.backend_pdf.PdfPages`` file."""
  import re
  with open(path, "rb") as f:
    data = f.read()
  return len(re.findall(rb"/Type\s*/Page(?!s)", data))


class TestDefaultLabels:

  def test_shares_no_suffix_uses_basenames(self, tmp_path):
    labels = compare._default_labels(str(tmp_path / "a"), str(tmp_path / "b"))
    assert labels == ["a", "b"]

  def test_identical_basenames_use_parent_to_disambiguate(self, tmp_path):
    labels = compare._default_labels(
        str(tmp_path / "baseline" / "sim"), str(tmp_path / "candidate" / "sim"))
    assert labels == [os.path.join("baseline", "sim"),
                      os.path.join("candidate", "sim")]


class TestParseLabelPair:

  def test_none_passes_through(self):
    assert compare._parse_label_pair(None) is None

  def test_valid_pair_is_split_and_stripped(self):
    assert compare._parse_label_pair("baseline,candidate") == [
        "baseline", "candidate"
    ]
    assert compare._parse_label_pair(" baseline , candidate ") == [
        "baseline", "candidate"
    ]

  @pytest.mark.parametrize("bad", ["onlyone", "a,b,c", "a,", ",b", ""])
  def test_malformed_pair_raises(self, bad):
    with pytest.raises(ValueError, match="exactly two comma-separated"):
      compare._parse_label_pair(bad)


class TestIsExcluded:

  def test_nothing_excluded_by_default(self):
    assert compare._DEFAULT_EXCLUDE_PREFIXES == ()
    assert not compare._is_excluded("geo_int_bmag",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("bc_zlower_vcutsq",
                                    compare._DEFAULT_EXCLUDE_PREFIXES)
    assert not compare._is_excluded("bflux_xlower_HamiltonianMoments",
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


class TestDominantAxes:
  """``_dominant_axes`` picks which axes to pin (beyond the 2 kept for
  plotting) from the data itself -- the point of largest relative
  difference, and which directions the difference changes fastest there."""

  def test_two_or_fewer_free_axes_pins_nothing(self):
    values0 = np.zeros((3, 4))
    values1 = np.ones((3, 4))
    grid = [np.linspace(0, 1, 3), np.linspace(0, 1, 4)]
    assert compare._dominant_axes(values0, values1, grid, [0, 1]) == {}
    assert compare._dominant_axes(values0, values1, grid, [0]) == {}

  def test_pins_the_flattest_axis_at_the_spike_location(self):
    # 3 free axes: axis 0 varies steeply around the spike, axis 1 is where
    # the spike sits, axis 2 is perfectly flat everywhere -- axis 2 should
    # always be the one pinned, regardless of where along it we look.
    rng = np.random.default_rng(0)
    shape = (6, 5, 4)
    values0 = np.ones(shape) * 10.0
    values1 = values0.copy()
    spike_idx = (3, 2, 1)
    values1[spike_idx] = 50.0
    # Make axis 0 steep near the spike so it's clearly favored over axis 1.
    ramp = np.linspace(0.0, 5.0, shape[0])
    values1 += ramp[:, None, None]
    grid = [np.linspace(0, 1, n) for n in shape]
    pins = compare._dominant_axes(values0, values1, grid, [0, 1, 2])
    assert len(pins) == 1
    assert 2 in pins  # the flat axis is the one pinned
    assert pins[2] == spike_idx[2]

  def test_near_zero_pivot_value_falls_back_to_raw_derivative(self):
    # The difference at its own point of max relative difference is tiny in
    # absolute terms (large relative diff from a tiny baseline), so scoring
    # must fall back to |derivative| instead of |derivative / value| to
    # avoid dividing by ~0.
    shape = (5, 5, 5)
    values0 = np.full(shape, 1e-10)
    values1 = values0.copy()
    values1[2, 2, 2] = 1e-8  # huge relative diff, but a tiny absolute value
    grid = [np.linspace(0, 1, n) for n in shape]
    pins = compare._dominant_axes(values0, values1, grid, [0, 1, 2])
    assert len(pins) == 1  # ran to completion, no division blow-up/crash

  def test_boundary_pivot_uses_one_sided_difference(self):
    # The point of max difference sits at axis 0's lower edge (index 0);
    # np.gradient's edge_order=2 one-sided stencil must still produce a
    # finite, usable derivative there instead of raising or returning nan.
    shape = (6, 6, 6)
    values0 = np.zeros(shape)
    values1 = values0.copy()
    values1[0, 3, 3] = 10.0
    grid = [np.linspace(0, 1, n) for n in shape]
    pins = compare._dominant_axes(values0, values1, grid, [0, 1, 2])
    assert len(pins) == 1
    for axis, idx in pins.items():
      assert np.isfinite(idx)

  def test_short_axis_falls_back_to_first_order_edge(self):
    # One free axis has only 2 points -- too short for edge_order=2 (needs
    # at least 3); _dominant_axes must still run, not raise.
    shape = (2, 5, 5)
    values0 = np.zeros(shape)
    values1 = values0.copy()
    values1[1, 2, 2] = 5.0
    grid = [np.linspace(0, 1, n) for n in shape]
    pins = compare._dominant_axes(values0, values1, grid, [0, 1, 2])
    assert len(pins) == 1

  def test_pivot_picks_largest_relative_not_absolute_difference(self):
    # A small baseline with a modest absolute change (huge relative change)
    # must win the pivot over a much larger absolute change on a huge
    # baseline (tiny relative change) -- confirms ranking is by relative,
    # not absolute, difference. Axes 0/1 are flat everywhere except at the
    # pivot, so whichever pivot wins pins both of them at its own index.
    shape = (4, 4, 4)
    values0 = np.full(shape, 1000.0)
    values1 = values0.copy()
    values1[0, 0, 0] = 1000.0 + 100.0  # +10% relative, +100 absolute
    values0[3, 3, 3] = 0.001
    values1[3, 3, 3] = 0.002  # +100% relative, +0.001 absolute
    grid = [np.linspace(0, 1, n) for n in shape]
    pins = compare._dominant_axes(values0, values1, grid, [0, 1, 2])
    assert len(pins) == 1
    pinned_axis, pinned_idx = next(iter(pins.items()))
    assert pinned_idx == 3  # the small-baseline (100% relative diff) spot


class TestSimsDynvectorOnly:

  def test_plots_every_shared_quantity_within_num_plots(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5, file_per_fig=True))
    assert set(report.plotted) == {"field_energy", "elc_integrated_moms"}
    assert report.only_in_0 == ()
    assert report.only_in_1 == ()
    assert report.skipped == {}
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.jpg"))

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
                    num_plots=1, file_per_fig=True))
    assert report.plotted == ("elc_integrated_moms",)

  def test_must_include_is_always_plotted(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field_energy"],
                    file_per_fig=True))
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

  def test_exclude_skips_matching_prefix(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    exclude=["elc_"]))
    assert "elc_integrated_moms" not in [d.quantity for d in report.compared]

  def test_custom_labels_used_in_legend(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    labels="before,after"))
    assert report.plotted

  def test_single_component_label_has_no_suffix(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field_energy"], num_plots=1,
                    labels="before,after", file_per_fig=True))
    fig = plt.gcf()
    lines = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert lines == ["before", "after"]

  def test_multi_component_label_gets_per_component_suffix(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["elc_integrated_moms"], num_plots=1,
                    labels="before,after", file_per_fig=True))
    fig = plt.gcf()
    lines = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert lines == [
        "before_c0", "after_c0", "before_c1", "after_c1", "before_c2",
        "after_c2"
    ]

  def test_format_controls_saved_extension(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    format="png", file_per_fig=True))
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.png"))


class TestSimsDiff:
  """``diff`` additionally plots ``baseline - candidate`` for 1-D/dynvector
  data: one more panel, appended after the baseline/candidate overlay
  panel(s), labeled with the two labels joined by ``" - "``."""

  def test_single_component_diff_adds_third_line_with_diff_label(
      self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", energy_scale=1.5)
    _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field_energy"], num_plots=1,
                    labels="before,after", file_per_fig=True, diff=True))
    fig = plt.gcf()
    assert len(fig.axes) == 2
    lines = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert lines == ["before", "after", "before - after"]

  def test_diff_line_values_are_the_elementwise_difference(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", energy_scale=1.5)
    _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field_energy"], num_plots=1,
                    labels="before,after", file_per_fig=True, diff=True))
    fig = plt.gcf()
    lines_by_label = {
        l.get_label(): l
        for ax in fig.axes for l in ax.get_lines()
    }
    before = lines_by_label["before"].get_ydata()
    after = lines_by_label["after"].get_ydata()
    diff = lines_by_label["before - after"].get_ydata()
    np.testing.assert_allclose(diff, before - after)

  def test_multi_component_diff_adds_one_diff_panel_per_component(
      self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["elc_integrated_moms"], num_plots=1,
                    labels="before,after", file_per_fig=True, diff=True))
    fig = plt.gcf()
    assert len(fig.axes) == 6  # 3 overlay panels + 3 diff panels
    lines = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert lines == [
        "before_c0", "after_c0", "before_c1", "after_c1", "before_c2",
        "after_c2", "before_c0 - after_c0", "before_c1 - after_c1",
        "before_c2 - after_c2"
    ]

  def test_diff_false_matches_omitting_diff(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report_omitted = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs1"),
                    must_include=["field_energy"], num_plots=1,
                    labels="before,after", file_per_fig=True))
    report_explicit = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs2"),
                    must_include=["field_energy"], num_plots=1,
                    labels="before,after", file_per_fig=True, diff=False))
    assert report_omitted.plotted == report_explicit.plotted
    fig = plt.gcf()
    assert len(fig.axes) == 1
    lines = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert lines == ["before", "after"]

  def test_multiblock_diff_adds_one_diff_panel_per_block(self, tmp_path):
    path0 = str(tmp_path / "baseline")
    path1 = str(tmp_path / "candidate")
    os.makedirs(path0, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for path, scale in ((path0, 1.0), (path1, 1.2)):
      for block in range(3):
        _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                     (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    report = _only(
        compare.sims(path0, path1, num_plots=1, out_dir=str(tmp_path / "out"),
                    diff=True))
    assert report.plotted == ("field_energy",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    # Title page + one page per block (3 blocks) -- diff adds a panel WITHIN
    # each block's page, not additional pages.
    assert _pdf_page_count(pdf_path) == 1 + 3


class TestSimsMultiblockDynvector:

  def _multiblock_sim(self, tmp_path, subdir, scale=1.0, num_blocks=3):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for block in range(num_blocks):
      _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                   (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    return path

  def test_plots_one_file_per_block(self, tmp_path):
    path0 = self._multiblock_sim(tmp_path, "baseline")
    path1 = self._multiblock_sim(tmp_path, "candidate", scale=1.2)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, file_per_fig=True))
    assert report.plotted == ("field_energy",)
    images = sorted(
        f for f in os.listdir(str(tmp_path / "figs")) if f.endswith(".jpg"))
    assert images == [
        "field_energy_b0.jpg", "field_energy_b1.jpg", "field_energy_b2.jpg"
    ]


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
                    num_plots=1, must_include=["field"], file_per_fig=True))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.jpg"))

  def _field_sim_3d(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    # serendipity p1 in 3D packs 8 basis coefficients per cell.
    values = rng.standard_normal((4, 4, 4, 8)) + offset
    _write_field(path, "sim", "field", [4, 4, 4], values, frame=0)
    return path

  def test_3d_field_plots_without_any_selectors(self, tmp_path):
    path0 = self._field_sim_3d(tmp_path, "baseline")
    path1 = self._field_sim_3d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"], file_per_fig=True))
    assert report.plotted == ("field",)
    assert os.path.isfile(os.path.join(str(tmp_path / "figs"), "field.jpg"))

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
    # serendipity p1 in 4D packs 16 basis coefficients per cell.
    values = rng.standard_normal((2, 2, 2, 2, 16)) + offset
    _write_field(path, "sim", "field", [2, 2, 2, 2], values, frame=0)
    return path

  def test_4d_field_plots_without_any_selectors_when_auto_filled(
      self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=5))
    assert "field" in report.plotted

  def test_4d_field_plots_without_any_selectors_as_must_include(
      self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field"]))
    assert report.plotted == ("field",)

  def test_4d_field_plots_with_explicit_selectors(self, tmp_path):
    path0 = self._field_sim_4d(tmp_path, "baseline")
    path1 = self._field_sim_4d(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    must_include=["field"], z1="0", z2="0"))
    assert report.plotted == ("field",)

  def _multiblock_field_sim(self, tmp_path, subdir, num_blocks=3, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    for block in range(num_blocks):
      values = rng.standard_normal((4, 4, 4)) + offset
      _write_field(path, f"sim_b{block}", "field", [4, 4], values, frame=0)
    return path

  def test_multiblock_field_contributes_one_page_per_block(self, tmp_path):
    path0 = self._multiblock_field_sim(tmp_path, "baseline")
    path1 = self._multiblock_field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, num_plots=1,
                    must_include=["field"], out_dir=str(tmp_path / "out")))
    assert report.plotted == ("field",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    # Title page + one page per block (3 blocks), no per-quantity image files.
    assert _pdf_page_count(pdf_path) == 1 + 3


@needs_gkeyll
class TestSimsDiffField:
  """``diff`` for a 2-D field: one more side-by-side panel, with its own
  independent color scale, appended after the baseline/candidate panels."""

  def _field_sim(self, tmp_path, subdir, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 4, 4)) + offset
    _write_field(path, "sim", "field", [4, 4], values, frame=0)
    return path

  def _multiblock_field_sim(self, tmp_path, subdir, num_blocks=3, offset=0.0):
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    for block in range(num_blocks):
      values = rng.standard_normal((4, 4, 4)) + offset
      _write_field(path, f"sim_b{block}", "field", [4, 4], values, frame=0)
    return path

  def _multicomp_field_sim(self, tmp_path, subdir, offset=0.0):
    """A 2-physical-component field: 2 interleaved p1 serendipity fields,
    each contributing 4 basis coefficients per cell (8 total)."""
    path = str(tmp_path / subdir)
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    nc_per_field = generate_test_data.num_comps("serendipity", 2, 1)
    values = rng.standard_normal((4, 4, 2 * nc_per_field)) + offset
    _write_field(path, "sim", "field", [4, 4], values, frame=0)
    return path

  @staticmethod
  def _data_axes(fig):
    """The figure's pcolormesh data panels, excluding the narrow (fixed 3%
    width) colorbar axes ``make_axes_locatable`` appends beside each one --
    filtering on ``.collections`` alone doesn't distinguish them, since a
    colorbar axis has its own ``.collections`` entry too."""
    return [ax for ax in fig.axes if ax.get_position().width > 0.05]

  def test_2d_diff_adds_third_panel_with_independent_scale(self, tmp_path):
    path0 = self._field_sim(tmp_path, "baseline")
    path1 = self._field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"], labels="before,after",
                    file_per_fig=True, diff=True))
    assert report.plotted == ("field",)
    fig = plt.gcf()
    data_axes = self._data_axes(fig)
    assert len(data_axes) == 3
    clims = [ax.collections[0].get_clim() for ax in data_axes]
    assert clims[0] == clims[1]
    assert clims[2] != clims[0]

  def test_multiblock_field_diff_contributes_one_diff_panel_per_block(
      self, tmp_path):
    path0 = self._multiblock_field_sim(tmp_path, "baseline")
    path1 = self._multiblock_field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, num_plots=1, must_include=["field"],
                    out_dir=str(tmp_path / "out"), diff=True))
    assert report.plotted == ("field",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    # Title page + one page per block (3 blocks) -- diff adds a panel WITHIN
    # each block's page, not additional pages.
    assert _pdf_page_count(pdf_path) == 1 + 3

  def test_multi_component_2d_diff_gets_per_component_labels(self, tmp_path):
    path0 = self._multicomp_field_sim(tmp_path, "baseline")
    path1 = self._multicomp_field_sim(tmp_path, "candidate", offset=0.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    num_plots=1, must_include=["field"], labels="before,after",
                    file_per_fig=True, diff=True))
    assert report.plotted == ("field",)
    fig = plt.gcf()
    data_axes = self._data_axes(fig)
    assert len(data_axes) == 6  # 2 components x (baseline, candidate, diff)
    texts = [ax.texts[0].get_text() for ax in data_axes]
    assert texts == [
        "before_c0", "before_c1", "after_c0", "after_c1",
        "before_c0 - after_c0", "before_c1 - after_c1"
    ]


class TestSimsBatch:

  def test_compares_every_shared_subdirectory(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    reports = compare.sims(str(base), str(cand), batch=True,
                           out_dir=str(tmp_path / "figs"),
                           file_per_fig=True)
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

  def test_labels_replace_basenames_for_every_pair(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    compare.sims(str(base), str(cand), batch=True,
                out_dir=str(tmp_path / "figs"), num_plots=1,
                must_include=["field_energy"], labels="before,after",
                file_per_fig=True)
    # Every pair reuses the same shared figure 0, so only the
    # last-plotted sim's labels remain on screen; checking that they are
    # exactly "before"/"after" (not that sim's basename) confirms labels
    # was honored rather than silently ignored, as it was before this
    # change.
    lines = [l.get_label() for ax in plt.gcf().axes for l in ax.get_lines()]
    assert lines == ["before", "after"]


class TestVerbose:
  """``verbose`` controls only how the returned value prints at the CLI
  edge (``repr()``, since the CLI echoes a command's return value verbatim)
  -- the value itself is always a fully-populated, ordinary-dict-compatible
  mapping regardless."""

  def test_default_repr_is_a_short_per_sim_summary(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                          must_include=["field_energy"], num_plots=1,
                          file_per_fig=True)
    text = repr(report)
    assert "baseline: 1 plotted, 0 skipped" in text
    assert "--verbose" in text
    assert "QuantityDiff" not in text

  def test_verbose_repr_is_the_full_dict(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                          must_include=["field_energy"], num_plots=1,
                          file_per_fig=True, verbose=True)
    text = repr(report)
    assert "QuantityDiff" in text
    assert "CompareSimsReport" in text

  def test_returned_value_behaves_like_a_plain_dict_either_way(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                          must_include=["field_energy"], num_plots=1,
                          file_per_fig=True)
    assert isinstance(report, dict)
    assert report == {"baseline": report["baseline"]}
    assert report["baseline"].plotted == ("field_energy",)

  def test_batch_summary_has_one_line_per_simulation(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    report = compare.sims(str(base), str(cand), batch=True,
                          out_dir=str(tmp_path / "figs"), num_plots=1,
                          must_include=["field_energy"], file_per_fig=True)
    text = repr(report)
    assert "sim_a:" in text
    assert "sim_b:" in text


class TestOutDirDefault:

  def test_out_dir_omitted_uses_cwd_subfolder(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(compare.sims(path0, path1, file_per_fig=True))
    expected = os.path.join(str(tmp_path), "postgkyl_compare_sims")
    assert report.out_dir == expected
    assert os.path.isfile(os.path.join(expected, "field_energy.jpg"))

  def test_out_dir_given_is_honored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "custom")))
    assert report.out_dir == str(tmp_path / "custom")


class TestCombinedPdf:
  """The default (no ``file_per_fig``) output mode: one combined,
  multi-page PDF instead of individual image files."""

  def test_defaults_to_cwd_filename(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(compare.sims(path0, path1))
    pdf_path = os.path.join(str(tmp_path), "postgkyl_compare_sims.pdf")
    assert os.path.isfile(pdf_path)
    # Title page + one page per plotted quantity; no individual image files.
    assert _pdf_page_count(pdf_path) == 1 + len(report.plotted)
    assert not os.path.isdir(
        os.path.join(str(tmp_path), "postgkyl_compare_sims"))

  def test_out_dir_given_no_pdf_name(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, out_dir=str(tmp_path / "out"))
    assert os.path.isfile(
        os.path.join(str(tmp_path / "out"), "postgkyl_compare_sims.pdf"))

  def test_pdf_name_given_no_out_dir(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, pdf_name="report.pdf")
    assert os.path.isfile(os.path.join(str(tmp_path), "report.pdf"))

  def test_pdf_name_and_out_dir_given(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, pdf_name="report.pdf",
                out_dir=str(tmp_path / "out"))
    assert os.path.isfile(os.path.join(str(tmp_path / "out"), "report.pdf"))

  def test_pdf_name_with_file_per_fig_raises(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    with pytest.raises(ValueError, match="pdf_name cannot be combined"):
      compare.sims(path0, path1, pdf_name="report.pdf", file_per_fig=True)

  def test_format_is_ignored(self, tmp_path):
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    compare.sims(path0, path1, format="png", out_dir=str(tmp_path / "out"))
    assert os.path.isfile(
        os.path.join(str(tmp_path / "out"), "postgkyl_compare_sims.pdf"))

  def test_multiblock_dynvector_contributes_one_page_per_block(self, tmp_path):
    path0 = str(tmp_path / "baseline")
    path1 = str(tmp_path / "candidate")
    os.makedirs(path0, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    time = np.linspace(0.0, 1.0, 4)
    for path, scale in ((path0, 1.0), (path1, 1.2)):
      for block in range(3):
        _write_dynvec(path, f"sim_b{block}", "field_energy", time,
                     (np.linspace(1.0, 2.0, 4) * scale * (block + 1))[:, None])
    report = _only(
        compare.sims(path0, path1, num_plots=1,
                    out_dir=str(tmp_path / "out")))
    assert report.plotted == ("field_energy",)
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    assert _pdf_page_count(pdf_path) == 1 + 3  # title page + one per block

  def test_batch_writes_one_combined_pdf(self, tmp_path):
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"
    for sim_name in ("sim_a", "sim_b"):
      _sim(base.parent, base.name + f"/{sim_name}", name=sim_name)
      _sim(cand.parent, cand.name + f"/{sim_name}", name=sim_name,
          moms_scale=1.5)
    reports = compare.sims(str(base), str(cand), batch=True,
                           out_dir=str(tmp_path / "out"))
    assert set(reports) == {"sim_a", "sim_b"}
    pdf_path = os.path.join(str(tmp_path / "out"),
                            "postgkyl_compare_sims.pdf")
    assert os.path.isfile(pdf_path)
    total_plotted = sum(len(r.plotted) for r in reports.values())
    assert _pdf_page_count(pdf_path) == 1 + total_plotted
    # No per-simulation subfolders when writing the combined PDF.
    assert not os.path.isdir(os.path.join(str(tmp_path / "out"), "sim_a"))
    assert not os.path.isdir(os.path.join(str(tmp_path / "out"), "sim_b"))


class TestFilePerFig:
  """The opt-in individual-image-files output mode."""

  def test_writes_individual_files_not_a_pdf(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path0 = _sim(tmp_path, "baseline")
    path1 = _sim(tmp_path, "candidate", moms_scale=1.5)
    report = _only(
        compare.sims(path0, path1, out_dir=str(tmp_path / "figs"),
                    file_per_fig=True))
    for quantity in report.plotted:
      assert os.path.isfile(
          os.path.join(str(tmp_path / "figs"), f"{quantity}.jpg"))
    assert not os.path.isfile(
        os.path.join(str(tmp_path), "postgkyl_compare_sims.pdf"))
    assert not os.path.isfile(
        os.path.join(str(tmp_path / "figs"), "postgkyl_compare_sims.pdf"))


class TestTitlePageFigure:

  def test_non_batch_names_both_paths(self):
    fig = compare._title_page_figure("/a/baseline", "/a/candidate",
                                     batch=False)
    text = fig.texts[0].get_text()
    assert "/a/baseline" in text
    assert "/a/candidate" in text

  def test_batch_lists_every_simulation(self):
    fig = compare._title_page_figure(
        "/a/baseline", "/a/candidate", batch=True,
        sim_names=["sim_a", "sim_b", "sim_c"])
    text = fig.texts[0].get_text()
    assert "/a/baseline" in text
    assert "/a/candidate" in text
    for name in ("sim_a", "sim_b", "sim_c"):
      assert name in text
