"""Contracts of the native setuptools command, independent of compilation."""

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
import setuptools
from setuptools.dist import Distribution

ROOT = Path(__file__).parents[1]


@pytest.fixture
def setup_config(monkeypatch):
  config = {}
  monkeypatch.delenv("POSTGKYL_GKEYLL_DIR", raising=False)
  monkeypatch.setattr(setuptools, "setup",
                      lambda **kwargs: config.update(kwargs))
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "0")
  namespace = runpy.run_path(ROOT / "setup.py")
  return config, namespace


# Setuptools asks install_lib for output metadata even for PEP 660 builds.
@pytest.mark.filterwarnings(
    "ignore:setup.py install is deprecated:setuptools.warnings.SetuptoolsDeprecationWarning"
)
@pytest.mark.parametrize("editable", [False, True])
def test_native_outputs_include_library_and_provenance(setup_config, tmp_path,
                                                       editable):
  config, _ = setup_config
  distribution = Distribution({
      **config, "packages": ["postgkyl.gpython"],
      "package_dir": {
          "": "src"
      }
  })
  command = distribution.get_command_obj("build_ext")
  command.build_lib = str(tmp_path / "lib")
  command.editable_mode = editable
  command.ensure_finalized()
  outputs = command.get_outputs()
  for name in ("libg0core.so", "_build_info.json"):
    built = str(tmp_path / "lib/postgkyl/gpython" / name)
    assert built in outputs
    assert command.get_output_mapping()[built] == str(
        Path("src/postgkyl/gpython") / name)
  assert any(Path(path).name.startswith("_gpython.") for path in outputs)
  assert distribution.has_ext_modules()


def test_skip_build_produces_pure_python_distribution(monkeypatch):
  config = {}
  monkeypatch.setattr(setuptools, "setup",
                      lambda **kwargs: config.update(kwargs))
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "1")
  runpy.run_path(ROOT / "setup.py")
  assert not Distribution(config).has_ext_modules()


def test_skip_build_rejects_misspelled_value(monkeypatch):
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "true")
  with pytest.raises(ValueError, match="must be 0 or 1"):
    runpy.run_path(ROOT / "setup.py")


@pytest.mark.parametrize("headers,library", [
    ("core/zero", "build/core/libg0core.so"),
    ("include", "lib/libg0core.so"),
])
def test_external_build_is_selected_without_building(setup_config, monkeypatch,
                                                     tmp_path, headers,
                                                     library):
  directory = tmp_path / "external build"
  include = directory / headers
  core = directory / library
  include.mkdir(parents=True)
  core.parent.mkdir(parents=True)
  (include / "gkyl_gpython.h").touch()
  core.touch()
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR", str(directory))
  with patch("subprocess.run") as run:
    selected = setup_config[1]["_gkeyll_paths"]()
  assert selected == (directory, include, core)
  run.assert_not_called()


@pytest.mark.parametrize("missing", ["header", "library", "both", "empty"])
def test_invalid_external_build_never_falls_back(setup_config, monkeypatch,
                                                 tmp_path, missing):
  include = tmp_path / "include"
  library = tmp_path / "lib"
  include.mkdir()
  library.mkdir()
  if missing == "header":
    (library / "libg0core.so").touch()
  if missing == "library":
    (include / "gkyl_gpython.h").touch()
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR",
                     "" if missing == "empty" else str(tmp_path))
  with patch("subprocess.run") as run:
    with pytest.raises(ValueError, match="POSTGKYL_GKEYLL_DIR"):
      setup_config[1]["_gkeyll_paths"]()
  run.assert_not_called()


def test_default_build_uses_internal_checkout(setup_config):
  with patch("subprocess.run") as run:
    selected = setup_config[1]["_gkeyll_paths"]()
  assert selected == (ROOT / "gkeyll", ROOT / "gkeyll/core/zero",
                      ROOT / "gkeyll/build/core/libg0core.so")
  run.assert_called_once_with(
      ["sh", str(ROOT / "scripts/build_gkeyll.sh")], cwd=ROOT, check=True)
