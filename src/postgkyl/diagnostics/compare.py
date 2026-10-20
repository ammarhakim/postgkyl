"""Equation-blind comparison of two Gkeyll simulations' output directories.

Finds the output quantities two simulation runs have in common, ranks them
by how much they differ (:func:`postgkyl.numerics.max_abs_rel_diff`), and
plots the most-different ones (plus any the caller always wants) baseline
overlaid on candidate for 1-D/dynvector data, or side by side for 2-D
(including a ``select()``-ed slice of 3-D+ data). Shares a home with
:mod:`postgkyl.diagnostics.discovery` -- like it, this performs no equation-
specific physics and belongs to no model family -- rather than under
``gk``/``vm``/``pkpm``/``mom``. By default every figure is collected into one
multi-page PDF, via ``matplotlib.backends.backend_pdf.PdfPages`` (a plain
Matplotlib facility, not a new dependency); ``file_per_fig`` opts into
individual image files instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from postgkyl import io
from postgkyl.gdata.load import load
from postgkyl.gdatastate import materialize_point_values
from postgkyl.numerics import max_abs_rel_diff
from postgkyl.operations.interpolate import interpolate
from postgkyl.operations.select import select
from postgkyl.render.matplotlib import plot

from . import discovery

_DEFAULT_SUBDIR = "postgkyl_compare_sims"
_DEFAULT_PDF_NAME = "postgkyl_compare_sims.pdf"

# No quantity-name prefix is excluded by default: geometry (``geo_*``),
# per-boundary (``bc_*``), and flux (``bflux_*``) outputs are ranked and
# plottable like any other quantity. ``--exclude`` lets a caller skip
# specific prefixes on a case-by-case basis.
_DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuantityDiff:
  """One shared quantity's difference ranking.

  Attributes:
    quantity: The bare quantity name (e.g. ``"field_energy"``,
      ``"elc_integrated_moms"`` -- the part of the filename after the sim
      name and block suffix; see :mod:`postgkyl.io.naming`).
    max_abs_diff: Largest absolute element-wise difference across every
      block/frame compared for this quantity.
    max_rel_diff: Largest relative element-wise difference, likewise.
    plottable: False when this quantity has more spatial dimensions than
      ``sims()`` can reduce to two by default (e.g. a full distribution
      function, 4-D+ even after the ``y``-midpoint default): still ranked,
      but skipped when auto-filling ``num_plots`` unless named in
      ``must_include``, where it instead raises so the caller can pin its
      remaining axes with ``z2``..``z5``.
  """

  quantity: str
  max_abs_diff: float
  max_rel_diff: float
  plottable: bool = True


@dataclass(frozen=True)
class CompareSimsReport:
  """What :func:`sims` found, ranked, and plotted.

  Attributes:
    compared: Every shared quantity actually diffed, ranked by
      ``max_rel_diff`` descending.
    plotted: The quantities a figure was saved for, in the order plotted.
    only_in_0: Quantities found only under ``path0``.
    only_in_1: Quantities found only under ``path1``.
    skipped: Shared quantities neither ranked nor plotted because loading or
      differencing them raised (e.g. a non-field, non-dynvector file like a
      multiblock ``block_topo``), mapped to the error message.
    out_dir: Directory the figures were written to (resolved, never
      ``None``, even when the caller left ``out_dir`` unset). By default
      (combined PDF), this is the directory the PDF was written to, not the
      PDF path itself.
  """

  compared: tuple[QuantityDiff, ...]
  plotted: tuple[str, ...]
  only_in_0: tuple[str, ...] = field(default_factory=tuple)
  only_in_1: tuple[str, ...] = field(default_factory=tuple)
  skipped: dict[str, str] = field(default_factory=dict)
  out_dir: str = ""


def _default_labels(path0: str, path1: str) -> list[str]:
  """The shortest distinguishing trailing path segments of ``path0``/
  ``path1`` (mirrors ``render.matplotlib._dataset_labels``) -- a bare
  basename is useless whenever both sims share a name, as a baseline and a
  candidate directory for the same simulation normally do."""
  parts0 = Path(os.path.normpath(path0)).parts
  parts1 = Path(os.path.normpath(path1)).parts
  depth = 1
  while (depth < len(parts0) and depth < len(parts1)
        and parts0[-depth:] == parts1[-depth:]):
    depth += 1
  return [os.path.join(*parts0[-depth:]), os.path.join(*parts1[-depth:])]


def _parse_label_pair(labels: str | None) -> list[str] | None:
  """Split ``"label0,label1"`` into the two labels ``sims()``'s ``-l``
  takes, or ``None`` unchanged so callers fall back to
  :func:`_default_labels`.

  A single comma-separated string for ``path0``/``path1``'s pair of labels,
  rather than a repeatable/JSON-array list option like ``plot``'s
  ``--legend_labels`` -- there are always exactly two labels here (one per
  compared sim), so one ``-l 'baseline,candidate'`` reads more directly than
  ``-l baseline -l candidate`` or a quoted JSON array would.
  """
  if labels is None:
    return None
  parts = labels.split(",")
  if len(parts) != 2 or not all(p.strip() for p in parts):
    raise ValueError(
        "labels must be exactly two comma-separated, non-empty labels, "
        f"e.g. 'baseline,candidate' (got {labels!r})")
  return [p.strip() for p in parts]


def _stems_by_quantity(path: str) -> dict[str, str]:
  """Map each distinct quantity name under ``path`` to one representative
  stem (block index stripped, since blocks of the same field compare as one
  quantity)."""
  out: dict[str, str] = {}
  for stem in discovery.find_output_stems("gkyl", path)["gkyl"]:
    name = io.parse_output_name(f"{stem}.gkyl")
    out.setdefault(name.quantity, stem)
  return out


def _is_excluded(quantity: str, exclude: tuple[str, ...]) -> bool:
  return any(quantity.startswith(prefix) for prefix in exclude)


def _block_paths(path: str, stem: str) -> list[str]:
  """Every per-block file path for ``stem`` (one entry for a single-block
  stem), sorted by block index."""
  name = io.parse_output_name(f"{stem}.gkyl")
  if name.block is None:
    return [os.path.join(path, f"{stem}.gkyl")]
  blocks = sorted(
      b for b in discovery.find_output_stems("gkyl", path)["gkyl"]
      if io.parse_output_name(f"{b}.gkyl").quantity == name.quantity)
  return [os.path.join(path, f"{b}.gkyl") for b in blocks]


def _latest_common_frame(path0: str, stem0: str, path1: str,
                         stem1: str) -> int | None:
  """The latest frame both sims wrote for a framed quantity family, or
  ``None`` when the family carries no frame index at all (e.g. a
  dynvector/time-series quantity, always one file with no trailing
  number)."""
  frames0 = discovery.available_frames(os.path.join(path0, f"{stem0}_"))
  frames1 = discovery.available_frames(os.path.join(path1, f"{stem1}_"))
  common = frames0 & frames1
  if not common and not frames0 and not frames1:
    return None
  if not common:
    raise ValueError(
        f"'{stem0}' has no frame in common between the two sims "
        f"(found {sorted(frames0)} vs {sorted(frames1)})")
  return max(common)


def _load_quantity(path: str, stem: str, frame: int | None):
  """Load (and block-combine, if multiblock) one quantity family as a single
  dataset or list of per-block datasets."""
  paths = _block_paths(path, stem)
  suffix = f"_{frame}.gkyl" if frame is not None else ".gkyl"
  files = [p.removesuffix(".gkyl") + suffix for p in paths]
  return [load(f) for f in files]


def _point_values(data):
  """Interpolate/materialize ``data`` into plain point values for both
  plotting and differencing, leaving an already-evaluated dataset (a
  dynvector, or anything with no DG basis) untouched."""
  if data.ctx.get("basis_type") and not data.is_interpolated:
    data = interpolate(data)
  return materialize_point_values(data)


def _default_slice_selectors(data, zs: tuple) -> dict[str, float]:
  """Default a 3-D+ field to its midpoint in ``y`` (coordinate direction 1)
  when the caller left ``z1`` unpinned -- Gkeyll's gyrokinetic config-space
  axis order is ``(x, y, z, ...)``, so this is what turns a 3-D field into
  the same ``(x, z)`` 2-D cross-section used when comparing CI runs by hand.
  A 4-D+ field (e.g. a full distribution function) still needs the caller
  to pin its remaining axes explicitly via ``z2``..``z5``; ``plot`` raises
  its own error if more than two axes survive."""
  if data.num_dims < 3 or zs[1] is not None:
    return {}
  lower, upper = data.bounds
  return {"z1": float(0.5 * (lower[1] + upper[1]))}


def _sliced(data, zs: tuple):
  """Apply the caller's ``z0``..``z5`` selectors, defaulting a 3-D field's
  unpinned ``y`` axis to its midpoint (see
  :func:`_default_slice_selectors`)."""
  kwargs = {
      f"z{d}": z
      for d, z in enumerate(zs)
      if z is not None and d < data.num_dims
  }
  kwargs.update(_default_slice_selectors(data, zs))
  return select(data, **kwargs) if kwargs else data


def _num_dims_after_slice(data, zs: tuple) -> int:
  """How many spatial dimensions ``data`` would have left after applying
  the caller's ``z0``..``z5`` plus the ``y``-midpoint default -- without
  actually materializing/selecting it, just to decide plottability."""
  num_dims = data.num_dims
  pinned = sum(1 for d in range(min(num_dims, 6)) if zs[d] is not None)
  if num_dims >= 3 and zs[1] is None:
    pinned += 1  # the y-midpoint default pins one more axis
  return num_dims - pinned


def _resolve_frame(frame: int | None, path0: str, stem0: str, path1: str,
                   stem1: str) -> int | None:
  """The caller's pinned ``frame``, or the latest one both sims wrote."""
  if frame is not None:
    return frame
  return _latest_common_frame(path0, stem0, path1, stem1)


def _diff_quantity(path0: str, path1: str, quantity: str, stem0: str,
                   stem1: str, frame: int | None,
                   zs: tuple) -> QuantityDiff:
  blocks0 = _load_quantity(path0, stem0, frame)
  blocks1 = _load_quantity(path1, stem1, frame)
  if len(blocks0) != len(blocks1):
    raise ValueError(
        f"'{quantity}' has {len(blocks0)} blocks under {path0!r} but "
        f"{len(blocks1)} under {path1!r}")
  max_abs = max_rel = 0.0
  plottable = True
  for data0, data1 in zip(blocks0, blocks1):
    plottable = plottable and _num_dims_after_slice(data0, zs) <= 2
    values0 = _point_values(data0).values
    values1 = _point_values(data1).values
    if values0.shape != values1.shape:
      raise ValueError(
          f"'{quantity}' has shape {values0.shape} under {path0!r} but "
          f"{values1.shape} under {path1!r}")
    abs_diff, rel_diff = max_abs_rel_diff(values0, values1)
    max_abs = max(max_abs, abs_diff)
    max_rel = max(max_rel, rel_diff) if rel_diff != float("inf") else max_rel
  return QuantityDiff(quantity, max_abs, max_rel, plottable)


def _save_or_collect(fig, out_path: str | None, pdf: PdfPages | None) -> None:
  """Either save ``fig`` to ``out_path`` (today's behavior, ``fig`` was
  already written by ``plot(..., saveas=out_path)`` and this is a no-op), or
  append it as one page of ``pdf`` and close it -- closing lets
  ``plot(..., figure=0)`` safely recreate a fresh figure 0 the next time
  it's called, confirmed by testing (a closed figure number is reused
  cleanly, with no stale axes/legend bleeding into the next quantity)."""
  if pdf is not None:
    pdf.savefig(fig)
    plt.close(fig)


def _plot_quantity(path0: str, path1: str, stem0: str, stem1: str,
                   frame: int | None, zs: tuple, labels: list[str],
                   title: str, out_path: str | None,
                   pdf: PdfPages | None) -> None:
  blocks0 = _load_quantity(path0, stem0, frame)
  blocks1 = _load_quantity(path1, stem1, frame)
  num_blocks = len(blocks0)

  if num_blocks > 1:
    # Each block is its own coordinate patch of the same decomposed domain
    # (or, for a dynvector, its own independent time series) -- overlaying
    # every block's baseline+candidate on one shared axis gets unreadable
    # past a handful of blocks (one legend entry per block per sim), so
    # every multiblock quantity, dynvector or spatial field alike, gets one
    # plot per block.
    for block_idx, (data0, data1) in enumerate(zip(blocks0, blocks1)):
      block_title = f"{title} (block {block_idx})"
      block_path = None
      if out_path is not None:
        block_path = f"{os.path.splitext(out_path)[0]}_b{block_idx}" \
            f"{os.path.splitext(out_path)[1]}"
      _plot_one_block(data0, data1, zs, labels, block_title, block_path, pdf)
    return

  _plot_one_block(blocks0[0], blocks1[0], zs, labels, title, out_path, pdf)


def _plot_one_block(data0, data1, zs: tuple, labels: list[str], title: str,
                    out_path: str | None, pdf: PdfPages | None) -> None:
  data0 = _sliced(_point_values(data0), zs)
  data1 = _sliced(_point_values(data1), zs)
  saveas = out_path if pdf is None else None
  # plot()'s explicit legend_labels are used verbatim on every component
  # subplot, with no added _cN suffix (a deliberate, documented contract
  # relied on elsewhere) -- so a multi-component quantity would otherwise
  # show the identical pair of labels on every one of its component panels,
  # with no way to tell which panel is which component. Its *default*
  # label path (triggered by omitting legend_labels) does add that suffix,
  # built from each dataset's own .label -- so for >1 component, set the
  # label directly and let plot() derive "<label>_c<i>" on its own; for a
  # single component there is nothing to disambiguate, so pass legend_labels
  # explicitly instead, keeping today's plain, unsuffixed labels.
  if data0.num_comps > 1:
    data0.label, data1.label = labels
    legend_kwargs = {}
  else:
    legend_kwargs = {"legend_labels": labels}
  # Both datasets carry no multiblock identity of their own (a single sim
  # directory's quantity, not one block of a decomposed one), so without a
  # shared target figure `plot` would treat them as two singleton families
  # and draw each on its own figure instead of overlaying/pairing them;
  # figure=0 forces one shared figure, clear=True keeps it from
  # accumulating axes left over from an earlier quantity plotted in the
  # same process (e.g. an earlier call from the same `sims()` run).
  if data0.num_dims <= 1:
    fig = plot(data0,
              data1,
              figure=0,
              clear=True,
              title=title,
              no_show=True,
              saveas=saveas,
              **legend_kwargs)
  else:
    fig = plot(data0,
              data1,
              num_axes=2 * data0.num_comps,
              figure=0,
              clear=True,
              title=title,
              no_show=True,
              saveas=saveas,
              **legend_kwargs)
  _save_or_collect(fig, out_path, pdf)


def _compare_one(
    path0: str,
    path1: str,
    *,
    out_dir: str,
    num_plots: int,
    must_include: list[str] | None,
    exclude: list[str] | None,
    labels: list[str] | None,
    z0: str | None,
    z1: str | None,
    z2: str | None,
    z3: str | None,
    z4: str | None,
    z5: str | None,
    format: str,
    frame: int | None,
    pdf: PdfPages | None = None,
    title_prefix: str = "",
) -> CompareSimsReport:
  """One single-sim comparison -- the body ``sims()`` runs once directly, or
  once per shared subdirectory when ``batch=True``.

  ``labels`` is the already-resolved two-item legend-label list (see
  :func:`_parse_label_pair`); ``None`` falls back to :func:`_default_labels`.
  ``pdf``, when given, collects every quantity's figure as a page instead of
  writing individual image files (``out_dir``/``format`` are then unused for
  plotting, though ``out_dir`` is still reported on the returned
  :class:`CompareSimsReport`). ``title_prefix`` is prepended to each page's
  title (e.g. ``"<sim_name>: "`` in batch mode).
  """
  exclude_prefixes = _DEFAULT_EXCLUDE_PREFIXES + tuple(exclude or ())
  must_include = list(must_include or [])
  resolved_labels = labels or _default_labels(path0, path1)
  zs = (z0, z1, z2, z3, z4, z5)

  quantities0 = _stems_by_quantity(path0)
  quantities1 = _stems_by_quantity(path1)
  shared = sorted(set(quantities0) & set(quantities1))
  shared = [q for q in shared if not _is_excluded(q, exclude_prefixes)]
  only_in_0 = sorted(set(quantities0) - set(quantities1))
  only_in_1 = sorted(set(quantities1) - set(quantities0))

  missing_must_include = [
      quantity for quantity in must_include
      if quantity not in quantities0 or quantity not in quantities1
  ]
  if missing_must_include:
    raise ValueError(
        "must_include quantity(s) not present in both sims: "
        f"{', '.join(sorted(missing_must_include))}")

  compared: list[QuantityDiff] = []
  skipped: dict[str, str] = {}
  for quantity in shared:
    stem0, stem1 = quantities0[quantity], quantities1[quantity]
    try:
      common_frame = _resolve_frame(frame, path0, stem0, path1, stem1)
      compared.append(
          _diff_quantity(path0, path1, quantity, stem0, stem1, common_frame,
                        zs))
    except Exception as exc:  # noqa: BLE001 -- see docstring: an explicit
      # must_include still raises (below); an auto-ranked quantity merely
      # doesn't participate in ranking. Not every ".gkyl" stem is a field or
      # dynvector `load()` can read (e.g. a multiblock `block_topo` file),
      # and a single such output has nothing to do with the health of the
      # rest of the comparison.
      if quantity in must_include:
        raise
      skipped[quantity] = str(exc)
  compared.sort(key=lambda d: d.max_rel_diff, reverse=True)

  # Every must_include quantity (even an unplottable one -- it then raises
  # inside _plot_quantity, pointing the caller at the axes they still need
  # to pin), then the highest-ranked plottable rest, until num_plots is
  # filled. A quantity with too many spatial dimensions to reduce by
  # default (e.g. a full distribution function) is silently skipped when
  # only auto-filling, rather than crashing an otherwise-routine comparison.
  selected = list(dict.fromkeys(must_include))
  for diff in compared:
    if len(selected) >= num_plots:
      break
    if diff.quantity not in selected and diff.plottable:
      selected.append(diff.quantity)

  if pdf is None:
    os.makedirs(out_dir, exist_ok=True)
  plotted: list[str] = []
  for quantity in selected:
    stem0, stem1 = quantities0[quantity], quantities1[quantity]
    common_frame = _resolve_frame(frame, path0, stem0, path1, stem1)
    out_path = None if pdf is not None else os.path.join(
        out_dir, f"{quantity}.{format}")
    _plot_quantity(path0, path1, stem0, stem1, common_frame, zs,
                  resolved_labels, f"{title_prefix}{quantity}", out_path, pdf)
    plotted.append(quantity)

  return CompareSimsReport(
      compared=tuple(compared),
      plotted=tuple(plotted),
      only_in_0=tuple(only_in_0),
      only_in_1=tuple(only_in_1),
      skipped=skipped,
      out_dir=out_dir,
  )


def _shared_subdirs(path0: str, path1: str) -> list[str]:
  """Simulation subdirectory names ``path0``/``path1`` share, sorted."""
  names0 = {entry.name for entry in os.scandir(path0) if entry.is_dir()}
  names1 = {entry.name for entry in os.scandir(path1) if entry.is_dir()}
  common = sorted(names0 & names1)
  if not common:
    raise ValueError(
        f"no simulation subdirectory is shared between {path0!r} and "
        f"{path1!r}")
  return common


def _batch(
    path0: str,
    path1: str,
    *,
    out_dir: str,
    num_plots: int,
    must_include: list[str] | None,
    exclude: list[str] | None,
    labels: list[str] | None,
    z0: str | None,
    z1: str | None,
    z2: str | None,
    z3: str | None,
    z4: str | None,
    z5: str | None,
    format: str,
    frame: int | None,
    pdf: PdfPages | None = None,
) -> dict[str, CompareSimsReport]:
  """``_compare_one`` on every simulation ``path0``/``path1`` share. ``pdf``,
  when given, is shared by every simulation's pages instead of each sim
  writing its own ``<out_dir>/<name>/`` subfolder of individual files.
  ``labels`` (already resolved, see :func:`_parse_label_pair`), when given,
  replaces the basenames of ``path0``/``path1`` as the shared pair of legend
  labels every compared simulation uses."""
  common = _shared_subdirs(path0, path1)
  batch_labels = labels or [os.path.basename(os.path.normpath(path0)),
                           os.path.basename(os.path.normpath(path1))]
  return {
      name: _compare_one(
          os.path.join(path0, name),
          os.path.join(path1, name),
          out_dir=out_dir if pdf is not None else os.path.join(
              out_dir, name),
          num_plots=num_plots,
          must_include=must_include,
          exclude=exclude,
          labels=batch_labels,
          z0=z0, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5,
          format=format,
          frame=frame,
          pdf=pdf,
          title_prefix=f"{name}: ",
      )
      for name in common
  }


def _title_page_figure(path0: str, path1: str, *, batch: bool,
                       sim_names: list[str] | None = None):
  """A plain text-only title page naming what a combined-PDF comparison
  covers, so the PDF is self-contained without its filename or the command
  that produced it."""
  fig = plt.figure(figsize=(8.5, 11))
  lines = ["Gkeyll simulation comparison", ""]
  if batch:
    lines += [f"Baseline directory: {path0}", f"Candidate directory: {path1}",
              "", f"Simulations compared ({len(sim_names or [])}):"]
    lines += [f"  - {name}" for name in sim_names or []]
  else:
    lines += [f"Baseline: {path0}", f"Candidate: {path1}"]
  fig.text(0.1, 0.9, "\n".join(lines), ha="left", va="top", fontsize=12,
           family="monospace", wrap=True)
  return fig


def sims(
    path0: str,
    path1: str,
    *,
    batch: bool = False,
    out_dir: str | None = None,
    file_per_fig: bool = False,
    pdf_name: str | None = None,
    num_plots: int = 5,
    must_include: list[str] | None = None,
    exclude: list[str] | None = None,
    labels: str | None = None,
    z0: str | None = None,
    z1: str | None = None,
    z2: str | None = None,
    z3: str | None = None,
    z4: str | None = None,
    z5: str | None = None,
    format: str = "png",
    frame: int | None = None,
) -> dict[str, CompareSimsReport]:
  """Compare two Gkeyll simulations' output and plot their biggest differences.

  Finds every output quantity both sims wrote, ranks the shared ones by
  :func:`postgkyl.numerics.max_abs_rel_diff` (largest relative difference
  first), and plots ``num_plots`` of them -- always including every
  ``must_include`` quantity. Use ``exclude`` to skip specific quantities or
  prefixes. 

  A 2-D field (or a 3-D+ field once ``z0``..``z5`` have reduced it to 2-D)
  is drawn with the two sims side by side on a shared color scale. A 3-D field
  whose ``z1`` (``y``) axis is not pinned via ``z1`` defaults to its midpoint
  coordinate. A quantity with more spatial dimensions than that default can
  reduce to two (e.g. a full distribution function) is still ranked, but
  skipped when auto-filling ``num_plots``; naming it in ``must_include``
  instead raises, pointing at the ``z2``..``z5`` selectors it still needs.

  By default, every plotted quantity is collected into one combined,
  PDF, whose path is ``<out_dir>/<pdf_name>`` if ``out_dir`` was given, else
  ``<cwd>/<pdf_name>``. Alternatively, passing ``file_per_fig`` yields
  individual image files.

  ``batch`` reinterprets ``path0``/``path1`` as directories with multiple
  simulations each (rather than having one simulation each): each holds one
  subdirectory per simulation. Subdirectories or simulations present in both
  are compared.

  Args:
    path0: the first simulation's output directory (w/o --batch) or a
      directory with one subdirectory per simulation (w/ --batch) to compare
      against --path1.
    path1: Path to data from the second simulation(s).
    batch: Compare every simulation two directories-of-sims share.
    pdf_name: Filename for the combined PDF; defaults to
      ``"postgkyl_compare_sims.pdf"``.
    out_dir: The path to the directory where the combined PDF is written to
      (defaults to ``<cwd>/``). With --file_per_fig this is the directory
      individual figures are written to (defaults to
      ``<cwd>/postgkyl_compare_sims``).
    file_per_fig: Write individual image files instead of one combined PDF.
    num_plots: Number of quantities to plot, including every
      ``must_include`` entry.
    must_include: Names of quantities (e.g. ``"field_energy"``) that must
      beplotted.
    exclude: Quantity-name prefixes to skip (e.g. ``"geo_"``, ``"bc_"``).
    labels: The two legend labels for ``path0``/``path1`` as one
      comma-separated string, e.g. ``"baseline,candidate"``; defaults to
      each path's shortest distinguishing path segment
    z0: Selector for coordinate direction 0.
    z1: Selector for coordinate direction 1.
    z2: Selector for coordinate direction 2.
    z3: Selector for coordinate direction 3.
    z4: Selector for coordinate direction 4.
    z5: Selector for coordinate direction 5.
    format: Output image file extension (e.g. ``"png"``, ``"jpg"``,
      ``"pdf"``, ``"svg"``) if ``file_per_fig`` is set.
    frame: Frame to compare for framed (per-timestep) quantities; defaults
      to the latest frame both sims wrote. Quantities with no frame index
      (dynvectors, single-snapshot fields) ignore this.

  Returns:
    A dict mapping each compared simulation's name to its
    :class:`CompareSimsReport`. Without ``batch``, this has exactly one
    entry, keyed by ``path0``'s basename. With ``batch``, one entry per
    simulation subdirectory shared by ``path0`` and ``path1``.

  Raises:
    ValueError: a ``must_include`` quantity is missing from either sim, has
      too many spatial dimensions to plot without further ``z2``..``z5``
      selectors, two compared files are structurally incompatible (block
      count or array shape mismatch), ``labels`` is not exactly two
      comma-separated, non-empty labels, ``pdf_name`` was given together
      with ``file_per_fig``, or (``batch`` only) ``path0`` and ``path1``
      share no simulation subdirectory.
  """
  if pdf_name is not None and file_per_fig:
    raise ValueError("pdf_name cannot be combined with file_per_fig")
  resolved_labels = _parse_label_pair(labels)

  def _run(effective_out_dir: str,
          pdf: PdfPages | None) -> dict[str, CompareSimsReport]:
    if batch:
      return _batch(path0, path1, out_dir=effective_out_dir,
                   num_plots=num_plots, must_include=must_include,
                   exclude=exclude, labels=resolved_labels, z0=z0, z1=z1,
                   z2=z2, z3=z3, z4=z4, z5=z5, format=format, frame=frame,
                   pdf=pdf)
    report = _compare_one(path0, path1, out_dir=effective_out_dir,
                         num_plots=num_plots, must_include=must_include,
                         exclude=exclude, labels=resolved_labels, z0=z0,
                         z1=z1, z2=z2, z3=z3, z4=z4, z5=z5, format=format,
                         frame=frame, pdf=pdf)
    return {os.path.basename(os.path.normpath(path0)): report}

  if not file_per_fig:
    pdf_dir = out_dir if out_dir is not None else os.getcwd()
    pdf_path = os.path.join(pdf_dir, pdf_name or _DEFAULT_PDF_NAME)
    os.makedirs(pdf_dir, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
      sim_names = _shared_subdirs(path0, path1) if batch else None
      pdf.savefig(
          _title_page_figure(path0, path1, batch=batch,
                             sim_names=sim_names))
      return _run(pdf_dir, pdf)

  effective_out_dir = (out_dir if out_dir is not None else
                      os.path.join(os.getcwd(), _DEFAULT_SUBDIR))
  return _run(effective_out_dir, None)


__all__ = ["CompareSimsReport", "QuantityDiff", "sims"]
