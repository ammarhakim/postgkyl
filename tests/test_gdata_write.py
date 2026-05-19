"""Tests for GData write helpers."""

from __future__ import annotations

import json
import sys
import types

import numpy as np

from postgkyl.data.gdata import GData


class _FakeStructuredGrid:
  def __init__(self, _x, _y, _z):
    self._point_data = {}

  def __setitem__(self, key, value):
    self._point_data[key] = value

  def save(self, file_name):
    with open(file_name, "w", encoding="utf-8") as fh:
      fh.write("fake-vts")


def _write_vts(tmp_path, stem, suffix, *, time=None, frame=None):
  grid = [np.array([0.0, 1.0, 2.0])]
  values = np.array([[1.0], [2.0]])

  data = GData()
  data.push(grid, values)
  if time is not None:
    data.ctx["time"] = time
  if frame is not None:
    data.ctx["frame"] = frame

  out = tmp_path / f"{stem}_{suffix:04d}.vts"
  data.write(out_name=str(out), extension="vts")
  return out


def test_write_vts_creates_and_updates_series_sidecar(tmp_path, monkeypatch):
  fake_module = types.SimpleNamespace(StructuredGrid=_FakeStructuredGrid)
  monkeypatch.setitem(sys.modules, "pyvista", fake_module)

  first_out = _write_vts(tmp_path, "solution", 1, time=0.25)
  second_out = _write_vts(tmp_path, "solution", 2, time=0.50)

  series_file = tmp_path / "solution.vts.series"
  assert first_out.exists()
  assert second_out.exists()
  assert series_file.exists()

  with open(series_file, "r", encoding="utf-8") as fh:
    series_data = json.load(fh)

  assert series_data["file-series-version"] == "1.0"
  assert series_data["files"] == [
      {"name": "solution_0001.vts", "time": 0.25},
      {"name": "solution_0002.vts", "time": 0.5},
  ]


def test_write_vts_series_uses_frame_then_default_time(tmp_path, monkeypatch):
  fake_module = types.SimpleNamespace(StructuredGrid=_FakeStructuredGrid)
  monkeypatch.setitem(sys.modules, "pyvista", fake_module)

  _write_vts(tmp_path, "framecase", 1, frame=7)
  _write_vts(tmp_path, "framecase", 2)

  series_file = tmp_path / "framecase.vts.series"
  with open(series_file, "r", encoding="utf-8") as fh:
    series_data = json.load(fh)

  assert series_data["files"] == [
      {"name": "framecase_0002.vts", "time": 0.0},
      {"name": "framecase_0001.vts", "time": 7.0},
  ]


def test_write_vts_series_rewrites_existing_entry_without_duplication(tmp_path, monkeypatch):
  fake_module = types.SimpleNamespace(StructuredGrid=_FakeStructuredGrid)
  monkeypatch.setitem(sys.modules, "pyvista", fake_module)

  _write_vts(tmp_path, "resample", 1, time=0.10)
  _write_vts(tmp_path, "resample", 2, time=0.20)
  _write_vts(tmp_path, "resample", 2, time=0.40)

  series_file = tmp_path / "resample.vts.series"
  with open(series_file, "r", encoding="utf-8") as fh:
    series_data = json.load(fh)

  assert series_data["files"] == [
      {"name": "resample_0001.vts", "time": 0.1},
      {"name": "resample_0002.vts", "time": 0.4},
  ]
