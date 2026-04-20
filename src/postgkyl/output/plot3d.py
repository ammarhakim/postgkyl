"""Module including custom Gkeyll plotting function"""
from __future__ import annotations

import subprocess
import tempfile
import time
from itertools import product
from typing import Tuple, TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from postgkyl.utils import input_parser
from postgkyl.data.idx_parser import idx_parser as parse_idx
from postgkyl.data.select import select as data_select
if TYPE_CHECKING:
  from postgkyl import GData
# end


def _apply_plot_style(style: str | None, rcParams: dict | None, diverging: bool,
    cmap: str | None, xkcd: bool, background: str = "dark",
    invert_cmap: bool = False) -> dict:
  """Apply plot styling to Matplotlib and return Plotly theme colors."""
  background_name = (background or "dark").strip().lower()

  if bool(style):
    plt.style.use(style)
  elif background_name == "light":
    plt.style.use("default")
  else:
    plt.style.use(f"{os.path.dirname(os.path.realpath(__file__)):s}/postgkyl.mplstyle")
  # end

  # Define Plotly theme colors for both light and dark backgrounds
  if background_name == "light":
    mpl.rcParams["figure.facecolor"] = "#ffffff"
    mpl.rcParams["axes.facecolor"] = "#ffffff"
    mpl.rcParams["savefig.facecolor"] = "#ffffff"
    mpl.rcParams["text.color"] = "#111111"
    mpl.rcParams["axes.labelcolor"] = "#111111"
    mpl.rcParams["xtick.color"] = "#111111"
    mpl.rcParams["ytick.color"] = "#111111"
    mpl.rcParams["axes.edgecolor"] = "#222222"
    mpl.rcParams["grid.color"] = "#b8b8b8"
    theme_colors = dict(
        paper_color="#ffffff",
        scene_color="#ffffff",
        text_color="#111111",
        grid_color="#b8b8b8",
        axis_line_color="#222222",
    )
  else:
    theme_colors = dict(
        paper_color="#000000",
        scene_color="#000000",
        text_color="#e6e6e6",
        grid_color="#2a3242",
        axis_line_color="#9aa3b2",
    )
  # end

  if bool(rcParams):
    for key in rcParams:
      mpl.rcParams[key] = rcParams[key]
    # end
  # end

  cmap_name = None
  if bool(cmap):
    cmap_name = cmap
  elif bool(diverging):
    cmap_name = "RdBu_r"
  else:
    cmap_name = "inferno"
  # end

  if cmap_name is not None:
    mpl.rcParams["image.cmap"] = cmap_name
  # end

  if invert_cmap:
    current_cmap = mpl.rcParams["image.cmap"]
    if current_cmap.endswith("_r"):
      mpl.rcParams["image.cmap"] = current_cmap[:-2]
    else:
      mpl.rcParams["image.cmap"] = f"{current_cmap}_r"
    # end
  # end

  if xkcd:
    plt.xkcd()
  # end

  return theme_colors


def _plotly_colorscale(cmap_name: str, n: int = 256):
  cmap = mpl.colormaps.get_cmap(cmap_name).resampled(n)
  xs = np.linspace(0.0, 1.0, n)
  colorscale = []
  for x, rgba in zip(xs, cmap(xs)):
    r, g, b, a = rgba
    colorscale.append([float(x), f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {float(a):.3f})"])
  # end
  return colorscale


def _finite_range(values: np.ndarray) -> tuple[float, float]:
  finite = np.isfinite(values)
  if np.any(finite):
    finite_values = values[finite]
    return float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
  # end
  return float("nan"), float("nan")


def _axis_range(values: np.ndarray, axis_range: tuple[float, float] | None,
    log_axis: bool = False) -> list[float] | None:
  if axis_range is None:
    lower, upper = _finite_range(values)
  else:
    lower, upper = axis_range
  # end

  if not np.isfinite(lower) or not np.isfinite(upper):
    return None
  # end

  if log_axis:
    lower = np.log10(max(lower, np.finfo(float).tiny))
    upper = np.log10(max(upper, np.finfo(float).tiny))
  # end

  if lower == upper:
    padding = 1.0 if lower == 0.0 else abs(lower) * 0.05
    lower -= padding
    upper += padding
  # end

  return [lower, upper]


def _log_colorbar_ticks(log_min: float, log_max: float, max_ticks: int = 8) -> tuple[list[float], list[str]]:
  if not np.isfinite(log_min) or not np.isfinite(log_max):
    return [], []
  # end

  lo = int(np.floor(log_min))
  hi = int(np.ceil(log_max))
  if hi < lo:
    hi = lo
  # end

  count = hi - lo + 1
  step = max(1, int(np.ceil(count / max_ticks)))
  tick_vals = list(range(lo, hi + 1, step))

  # Ensure the upper bound appears as a tick label.
  if tick_vals[-1] != hi:
    tick_vals.append(hi)
  # end

  tick_text = [f"10<sup>{val:d}</sup>" for val in tick_vals]
  return [float(v) for v in tick_vals], tick_text


def _resolve_plotly_aspect(aspect: str | float | None, fixaspect: bool) -> tuple[str, dict | None]:
  if aspect is None:
    return ("cube", None) if fixaspect else ("auto", None)
  # end

  if isinstance(aspect, str):
    aspect_value = aspect.strip().lower()
    if aspect_value in ("auto", "data", "cube"):
      return aspect_value, None
    # end
    ratio = float(aspect)
    return "manual", dict(x=ratio, y=ratio, z=ratio)
  # end

  ratio = float(aspect)
  return "manual", dict(x=ratio, y=ratio, z=ratio)


def save_rotating_plotly_figure(fig, file_name: str,
    starting_azimuthal_angle: float, fps: int, polar_angle: float,
    rotation_period: float, radius: float = 2.0) -> None:
  """Save a rotating Plotly 3D figure as GIF or MP4.

  Rotates the camera 360 degrees around the vertical axis, starting from
  ``starting_azimuthal_angle`` in degrees.
  """
  root, ext = os.path.splitext(file_name)
  ext = ext.lower()
  if ext not in (".gif", ".mp4", ".html"):
    raise ValueError("--save-rotating expects an output ending with .gif, .mp4, or .html")
  # end
  if fps <= 0:
    raise ValueError("fps must be a positive integer")
  # end
  if rotation_period <= 0:
    raise ValueError("rotation_period must be positive")
  # end

  scene_names = [name for name in fig.layout.to_plotly_json().keys() if name == "scene" or name.startswith("scene")]
  if not scene_names:
    raise ValueError("Rotating export requires a Plotly 3D scene figure")
  # end
  scene_name = scene_names[0]

  polar_rad = np.deg2rad(polar_angle)
  xy_radius = radius * np.sin(polar_rad)
  z_eye = radius * np.cos(polar_rad)

  if ext == ".html":
    theta0 = np.deg2rad(starting_azimuthal_angle)
    initial_camera = dict(
        eye=dict(x=float(xy_radius * np.cos(theta0)), y=float(xy_radius * np.sin(theta0)), z=float(z_eye)),
        up=dict(x=0.0, y=0.0, z=1.0),
        center=dict(x=0.0, y=0.0, z=0.0),
    )
    fig.update_layout(**{scene_name: dict(camera=initial_camera)})

    omega = 2.0 * np.pi / float(rotation_period)

    if omega > 0.0:
      post_script = f"""
const gd = document.getElementById('{{plot_id}}');
const sceneName = '{scene_name}';
const defaultAzimuthDeg = {float(starting_azimuthal_angle):.17g};
const defaultPolarDeg = {float(polar_angle):.17g};
const defaultPeriodSec = {float(rotation_period):.17g};
const defaultRadius = {float(radius):.17g};
let rafId = null;
let startMs = null;

let azimuthDeg = defaultAzimuthDeg;
let polarDeg = defaultPolarDeg;
let periodSec = defaultPeriodSec;
let cameraRadius = defaultRadius;

let theta0 = 0.0;
let omega = 0.0;
let xyRadius = 0.0;
let zEye = 0.0;

const clampPositive = (value, fallback) => (Number.isFinite(value) && value > 0.0 ? value : fallback);

const recomputeRotationParams = () => {{
  const polarRad = polarDeg * Math.PI / 180.0;
  theta0 = azimuthDeg * Math.PI / 180.0;
  xyRadius = cameraRadius * Math.sin(polarRad);
  zEye = cameraRadius * Math.cos(polarRad);
  omega = 2.0 * Math.PI / periodSec;
}};

const updateCamera = (theta) => {{
  const camera = {{
    eye: {{x: xyRadius * Math.cos(theta), y: xyRadius * Math.sin(theta), z: zEye}},
    up: {{x: 0.0, y: 0.0, z: 1.0}},
    center: {{x: 0.0, y: 0.0, z: 0.0}}
  }};
  Plotly.relayout(gd, {{ [sceneName + '.camera']: camera }});
}};

const startRotation = () => {{
  if (rafId === null) {{
    rafId = requestAnimationFrame(animate);
  }}
}};

const stopRotation = () => {{
  if (rafId !== null) {{
    cancelAnimationFrame(rafId);
    rafId = null;
  }}
}};

const resetRotation = () => {{
  startMs = null;
  updateCamera(theta0);
  startRotation();
}};

const parent = gd.parentNode;
if (parent) {{
  if (getComputedStyle(parent).position === 'static') {{
    parent.style.position = 'relative';
  }}

  const controls = document.createElement('div');
  controls.style.position = 'absolute';
  controls.style.top = '12px';
  controls.style.left = '12px';
  controls.style.zIndex = '20';
  controls.style.background = 'rgba(255, 255, 255, 0.92)';
  controls.style.border = '1px solid #b7bec8';
  controls.style.borderRadius = '8px';
  controls.style.padding = '8px 10px';
  controls.style.fontFamily = 'sans-serif';
  controls.style.fontSize = '12px';
  controls.style.color = '#1f2933';
  controls.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.18)';
  controls.style.display = 'grid';
  controls.style.gridTemplateColumns = 'auto auto';
  controls.style.gap = '6px 8px';
  controls.style.alignItems = 'center';
  controls.style.opacity = '0';
  controls.style.pointerEvents = 'none';
  controls.style.transition = 'opacity 120ms ease';

  const showControlsButton = document.createElement('button');
  showControlsButton.type = 'button';
  showControlsButton.textContent = 'Show rotation controls';
  showControlsButton.style.position = 'absolute';
  showControlsButton.style.top = '12px';
  showControlsButton.style.left = '12px';
  showControlsButton.style.zIndex = '21';
  showControlsButton.style.fontSize = '12px';
  showControlsButton.style.padding = '4px 8px';
  showControlsButton.style.cursor = 'pointer';
  showControlsButton.style.opacity = '0';
  showControlsButton.style.pointerEvents = 'none';
  showControlsButton.style.transition = 'opacity 120ms ease';

  const makeNumberInput = (value, min, step) => {{
    const input = document.createElement('input');
    input.type = 'number';
    input.value = String(value);
    input.min = String(min);
    input.step = String(step);
    input.style.width = '86px';
    input.style.fontSize = '12px';
    return input;
  }};

  const addRow = (labelText, inputEl) => {{
    const label = document.createElement('label');
    label.textContent = labelText;
    controls.appendChild(label);
    controls.appendChild(inputEl);
  }};

  const periodInput = makeNumberInput(defaultPeriodSec, 0.001, 0.1);
  const azimuthInput = makeNumberInput(defaultAzimuthDeg, -3600, 1);
  const polarInput = makeNumberInput(defaultPolarDeg, -3600, 1);
  const radiusInput = makeNumberInput(defaultRadius, 0.001, 0.1);

  addRow('Period (s)', periodInput);
  addRow('Azimuth (deg)', azimuthInput);
  addRow('Polar (deg)', polarInput);
  addRow('Radius', radiusInput);

  const buttonWrap = document.createElement('div');
  buttonWrap.style.gridColumn = '1 / span 2';
  buttonWrap.style.display = 'flex';
  buttonWrap.style.gap = '8px';

  const applyButton = document.createElement('button');
  applyButton.type = 'button';
  applyButton.textContent = 'Apply';

  const stopButton = document.createElement('button');
  stopButton.type = 'button';
  stopButton.textContent = 'Stop rotation';

  const hideButton = document.createElement('button');
  hideButton.type = 'button';
  hideButton.textContent = 'Hide controls';

  for (const btn of [applyButton, stopButton, hideButton]) {{
    btn.style.fontSize = '12px';
    btn.style.padding = '3px 8px';
    btn.style.cursor = 'pointer';
  }}

  let controlsCollapsed = true;
  let hoverActive = false;
  let hideTimer = null;

  const setControlsVisible = (visible) => {{
    controls.style.opacity = visible ? '1' : '0';
    controls.style.pointerEvents = visible ? 'auto' : 'none';
  }};

  const setShowButtonVisible = (visible) => {{
    showControlsButton.style.opacity = visible ? '1' : '0';
    showControlsButton.style.pointerEvents = visible ? 'auto' : 'none';
  }};

  const refreshControlsVisibility = () => {{
    if (!hoverActive) {{
      setControlsVisible(false);
      setShowButtonVisible(false);
      return;
    }}
    if (controlsCollapsed) {{
      setControlsVisible(false);
      setShowButtonVisible(true);
    }} else {{
      setControlsVisible(true);
      setShowButtonVisible(false);
    }}
  }};

  const clearHideTimer = () => {{
    if (hideTimer !== null) {{
      clearTimeout(hideTimer);
      hideTimer = null;
    }}
  }};

  const scheduleHide = () => {{
    clearHideTimer();
    hideTimer = setTimeout(() => {{
      hoverActive = false;
      refreshControlsVisibility();
    }}, 100);
  }};

  const applyInputs = () => {{
    periodSec = clampPositive(parseFloat(periodInput.value), defaultPeriodSec);
    cameraRadius = clampPositive(parseFloat(radiusInput.value), defaultRadius);
    azimuthDeg = Number.isFinite(parseFloat(azimuthInput.value)) ? parseFloat(azimuthInput.value) : defaultAzimuthDeg;
    polarDeg = Number.isFinite(parseFloat(polarInput.value)) ? parseFloat(polarInput.value) : defaultPolarDeg;

    periodInput.value = String(periodSec);
    radiusInput.value = String(cameraRadius);
    azimuthInput.value = String(azimuthDeg);
    polarInput.value = String(polarDeg);

    recomputeRotationParams();
    resetRotation();
  }};

  applyButton.addEventListener('click', () => {{
    applyInputs();
  }});

  stopButton.addEventListener('click', () => {{
    stopRotation();
  }});

  hideButton.addEventListener('click', () => {{
    controlsCollapsed = true;
    refreshControlsVisibility();
  }});

  showControlsButton.addEventListener('click', () => {{
    controlsCollapsed = false;
    hoverActive = true;
    refreshControlsVisibility();
  }});

  parent.addEventListener('mouseenter', () => {{
    hoverActive = true;
    clearHideTimer();
    refreshControlsVisibility();
  }});

  parent.addEventListener('mouseleave', () => {{
    scheduleHide();
  }});

  buttonWrap.appendChild(applyButton);
  buttonWrap.appendChild(stopButton);
  buttonWrap.appendChild(hideButton);
  controls.appendChild(buttonWrap);
  parent.appendChild(controls);
  parent.appendChild(showControlsButton);
  refreshControlsVisibility();
}}

gd.addEventListener('mousedown', stopRotation);
gd.addEventListener('wheel', stopRotation);
gd.addEventListener('touchstart', stopRotation);

const animate = (timestamp) => {{
  if (startMs === null) {{
    startMs = timestamp;
  }}
  const elapsedSeconds = (timestamp - startMs) / 1000.0;
  const theta = theta0 + omega * elapsedSeconds;
  updateCamera(theta);
  rafId = requestAnimationFrame(animate);
}};

recomputeRotationParams();
updateCamera(theta0);
startRotation();
"""
      fig.write_html(file_name, include_plotlyjs="cdn", post_script=post_script)
    else:
      fig.write_html(file_name)
    # end
    return
  # end

  with tempfile.TemporaryDirectory(prefix="pgkyl_rotate_") as tmp_dir:
    output_label = os.path.basename(file_name) or file_name

    def _format_duration(seconds: float) -> str:
      total = max(0, int(round(seconds)))
      hrs, rem = divmod(total, 3600)
      mins, secs = divmod(rem, 60)
      if hrs > 0:
        return f"{hrs:d}:{mins:02d}:{secs:02d}"
      # end
      return f"{mins:02d}:{secs:02d}"

    def _print_progress(current: int, total: int, start_time: float) -> None:
      progress = current / max(1, total)
      elapsed = time.perf_counter() - start_time
      rate = current / elapsed if elapsed > 0 else 0.0
      remaining = (total - current) / rate if rate > 0 else float("inf")
      bar_width = 28
      filled = int(round(progress * bar_width))
      filled = min(bar_width, max(0, filled))
      bar = "#" * filled + "-" * (bar_width - filled)
      etr_text = _format_duration(remaining) if np.isfinite(remaining) else "--:--"
      print(
          f"\rRendering {output_label} [{bar}] {100.0 * progress:3.0f}% | {current:d} / {total:d} | ETR {etr_text}",
          end="",
          flush=True,
      )

    frame_pattern = os.path.join(tmp_dir, "frame_%05d.png")
    num_frames = max(2, int(round(float(fps) * float(rotation_period))))
    render_start = time.perf_counter()
    _print_progress(0, num_frames, render_start)
    for idx in range(num_frames):
      theta = np.deg2rad(
          starting_azimuthal_angle + 360.0 * idx / num_frames
      )
      camera = dict(
          eye=dict(x=float(xy_radius * np.cos(theta)), y=float(xy_radius * np.sin(theta)), z=float(z_eye)),
          up=dict(x=0.0, y=0.0, z=1.0),
          center=dict(x=0.0, y=0.0, z=0.0),
      )
      fig.update_layout(**{scene_name: dict(camera=camera) for scene_name in scene_names})
      png_bytes = fig.to_image(format="png")

      frame_path = os.path.join(tmp_dir, f"frame_{idx:05d}.png")
      with open(frame_path, "wb") as frame_file:
        frame_file.write(png_bytes)
      # end
      _print_progress(idx + 1, num_frames, render_start)
    # end
    print()

    if ext == ".mp4":
      ffmpeg_cmd = [
          "ffmpeg",
          "-y",
          "-framerate",
          str(fps),
          "-i",
          frame_pattern,
          "-pix_fmt",
          "yuv420p",
          file_name,
      ]
    else:
      ffmpeg_cmd = [
          "ffmpeg",
          "-y",
          "-framerate",
          str(fps),
          "-i",
          frame_pattern,
          "-vf",
          "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
          file_name,
      ]
    # end

    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  # end


def _prepare_3d_coordinates(coords: list[np.ndarray], value_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  arrays = tuple(np.asarray(coord) for coord in coords)
  if len(arrays) != 3:
    raise ValueError("Plotly 3D plotting requires exactly three coordinate arrays")
  # end
  if all(array.ndim == 1 for array in arrays):
    mesh = np.meshgrid(*arrays, indexing="ij")
    return mesh[0], mesh[1], mesh[2]
  # end
  if all(array.shape == value_shape for array in arrays):
    return arrays[0], arrays[1], arrays[2]
  # end
  return arrays[0], arrays[1], arrays[2]


def _resolve_slice_plane_index(axis_grid: np.ndarray, selector: int | float, axis_cells: int) -> int:
  axis_values = np.asarray(axis_grid)
  if axis_values.ndim == 1:
    len_grid = axis_values.shape[0]
  else:
    len_grid = axis_cells
  # end

  is_matching = axis_cells == len_grid
  axis_index = parse_idx(selector, axis_values, is_matching)
  if not isinstance(axis_index, int):
    raise TypeError("Slice selectors must resolve to a single axis index")
  # end

  if axis_index < 0:
    axis_index = axis_cells + axis_index
  # end
  if axis_index < 0 or axis_index >= axis_cells:
    raise IndexError(f"Slice selector index {axis_index:d} is out of range for axis size {axis_cells:d}")
  # end
  return axis_index


def _downsample_3d_volume(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    value: np.ndarray,
    maximum_points_per_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Downsample 3D arrays so no axis exceeds the configured maximum."""
  if value.ndim != 3:
    return x, y, z, value
  # end

  if maximum_points_per_axis is None or maximum_points_per_axis <= 0:
    return x, y, z, value
  # end

  steps = [max(1, int(np.ceil(size / maximum_points_per_axis))) for size in value.shape]
  if max(steps) == 1:
    return x, y, z, value
  # end

  def _axis_indices(size: int, step: int) -> np.ndarray:
    idx = np.arange(0, size, step, dtype=int)
    if idx[-1] != size - 1:
      idx = np.append(idx, size - 1)
    # end
    return idx

  idx0 = _axis_indices(value.shape[0], steps[0])
  idx1 = _axis_indices(value.shape[1], steps[1])
  idx2 = _axis_indices(value.shape[2], steps[2])

  def _take_indices(arr: np.ndarray) -> np.ndarray:
    out = np.take(arr, idx0, axis=0)
    out = np.take(out, idx1, axis=1)
    out = np.take(out, idx2, axis=2)
    return out

  return _take_indices(x), _take_indices(y), _take_indices(z), _take_indices(value)


def _latex_to_html(text: str) -> str:
  """Convert LaTeX subscripts and Greek letters to HTML."""
  if not text:
    return text
  text = text.strip()
  # Remove outer $ signs if present
  if text.startswith("$") and text.endswith("$"):
    text = text[1:-1]
  # Map common LaTeX commands to Unicode/HTML
  latex_to_unicode = {
      r'\mu': 'μ',
      r'\nu': 'ν',
      r'\pi': 'π',
      r'\sigma': 'σ',
      r'\Sigma': 'Σ',
      r'\rho': 'ρ',
      r'\tau': 'τ',
      r'\chi': 'χ',
      r'\phi': 'φ',
      r'\psi': 'ψ',
      r'\omega': 'ω',
      r'\Omega': 'Ω',
      r'\alpha': 'α',
      r'\beta': 'β',
      r'\gamma': 'γ',
      r'\delta': 'δ',
      r'\Delta': 'Δ',
      r'\epsilon': 'ε',
      r'\zeta': 'ζ',
      r'\eta': 'η',
      r'\theta': 'θ',
      r'\Theta': 'Θ',
      r'\iota': 'ι',
      r'\kappa': 'κ',
      r'\lambda': 'λ',
      r'\Lambda': 'Λ',
      r'\parallel': '∥',
      r'\perp': '⊥',
  }

  def _replace_latex_commands(value: str) -> str:
    for latex, unicode_char in latex_to_unicode.items():
      value = value.replace(latex, unicode_char)
    # end
    return value

  import re
  # Convert braced subscripts: _{...} -> <sub>...</sub>
  text = re.sub(
      r'_\{([^{}]+)\}',
      lambda match: f"<sub>{_replace_latex_commands(match.group(1))}</sub>",
      text,
  )
  # Convert unbraced subscripts: _x or _\parallel -> <sub>x</sub>/<sub>∥</sub>
  text = re.sub(
      r'_(\\[A-Za-z]+|[A-Za-z0-9])',
      lambda match: f"<sub>{_replace_latex_commands(match.group(1))}</sub>",
      text,
  )
  # Convert remaining LaTeX commands outside subscripts.
  text = _replace_latex_commands(text)
  return text


def _get_nodal_grid(grid : list, cells: np.ndarray):
  num_dims = len(grid)
  grid_out = []
  if num_dims != len(cells):  # sanity check
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  # end
  for d in range(num_dims):
    if len(grid[d].shape) == 1:
      if grid[d].shape[0] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[0] == cells[d] + 1:
        grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
      else:
        raise ValueError("Something is terribly wrong...")
      # end
    else:
      if grid[d].shape[d] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[d] == cells[d] + 1:
        if num_dims == 1:
          grid_out.append(0.5 * (grid[d][:-1] + grid[d][1:]))
        else:
          cell_shape = tuple(int(s - 1) for s in grid[d].shape)
          grid_avg = np.zeros(cell_shape, dtype=np.result_type(grid[d], float))
          for offset in product((0, 1), repeat=num_dims):
            sl = tuple(slice(o, o + cell_shape[i]) for i, o in enumerate(offset))
            grid_avg += grid[d][sl]
          # end
          grid_out.append(grid_avg / (2 ** num_dims))
        # end
      else:
        raise ValueError("Something is terribly wrong...")
      # end
    # end
  # end
  return grid_out


def plot3d(data: GData | Tuple[list, np.ndarray],
    squeeze: bool = False, num_axes: int = None,
    num_subplot_row: int | None = None, num_subplot_col: int | None = None,
    diverging: bool = False,
    xscale: float = 1.0, xshift: float = 0.0,
    yscale: float = 1.0, yshift: float = 0.0,
    zmin: float | None = None, zmax: float | None = None, zscale: float = 1.0, zshift: float = 0.0,
    cmin: float | None = None, cmax: float | None = None, cscale: float = 1.0, cshift: float = 0.0,
    clim: tuple[float, float] | None = None,
    style: str | None = None, rcParams: dict | None = None,
    background: str = "dark", invert_cmap: bool = False,
    legend: bool = True, label_prefix: str = "", colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None, clabel: str | None = None, title: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False, logc: bool = False,
    fixaspect: bool = False, aspect: str | float | None = None,
    showgrid: bool = True, hashtag: bool = False, xkcd: bool = False,
    color: str | None = None,
    linewidth: float | None = None, opacity: float | None = 1.0,
    maximum_points_per_axis: int = 0,
    surface_count: int = 32,
    xrange: tuple[float, float] | None = None, yrange: tuple[float, float] | None = None,
    zrange: tuple[float, float] | None = None,
    slice_plane: dict[str, int | float | list[int | float] | tuple[int | float, ...]] | None = None,
    figsize: tuple | None = None,
    cylindrical_to_cartesian: bool = False,
    cmap: str | None = None):
  """Plots 3D Gkeyll data using Plotly."""

  if go is None or make_subplots is None:
    raise ImportError("Plotly is required for 3D plots")
  # end

  theme_colors = _apply_plot_style(style, rcParams, diverging, cmap, xkcd, background=background,
      invert_cmap=invert_cmap)

  grid_in, values = input_parser(data)
  grid = grid_in.copy()

  if isinstance(data, tuple):
    if len(grid) == len(values.shape):
      num_dims = len(values.squeeze().shape)
    else:
      num_dims = len(values[..., 0].squeeze().shape)
    # end
    lg = len(grid)
    cells = np.zeros(lg)
    for d in range(lg):
      if len(grid[d].shape) == 1:
        cells[d] = len(grid[d])
      else:
        cells[d] = len(grid[d][d])
      # end
    # end
  else:
    num_dims = data.get_num_dims(squeeze=True)
    cells = data.get_num_cells()
  # end

  if num_dims != 3:
    raise ValueError("Plot3d handles only 3D data")
  # end

  axes_labels = ["$z_0$", "$z_1$", "$z_2$", "$z_3$", "$z_4$", "$z_5$"]
  if len(grid) > num_dims:
    idx = []
    for dim, g in enumerate(grid):
      if cells[dim] <= 1:
        idx.append(dim)
      # end
      grid[dim] = g.squeeze()
    # end
    if bool(idx):
      for i in reversed(idx):
        grid.pop(i)
      # end
      cells = np.delete(cells, idx)
      axes_labels = np.delete(axes_labels, idx)
      values = np.squeeze(values, tuple(idx))
      if len(grid[0].shape) > 1:
        for d in range(num_dims):
          for i in reversed(idx):
            grid[d] = np.mean(grid[d], axis=i)
          # end
        # end
      # end
    # end
  # end

  num_comps = values.shape[-1]
  idx_comps = range(num_comps)
  if num_axes:
    num_comps = num_axes
  else:
    num_comps = len(idx_comps)
  # end

  if xlabel is None:
    xlabel = "$x$" if cylindrical_to_cartesian else axes_labels[0]
    if xshift != 0.0 and xscale != 1.0:
      xlabel = rf"({xlabel:s} + {xshift:.2e}) $\times$ {xscale:.2e}"
    elif xshift != 0.0:
      xlabel = rf"{xlabel:s} + {xshift:.2e}"
    elif xscale != 1.0:
      xlabel = rf"{xlabel:s} $\times$ {xscale:.2e}"
    # end
  # end
  if ylabel is None:
    ylabel = "$y$" if cylindrical_to_cartesian else axes_labels[1]
    if yshift != 0.0 and yscale != 1.0:
      ylabel = rf"({ylabel:s} + {yshift:.2e}) $\times$ {yscale:.2e}"
    elif yshift != 0.0:
      ylabel = rf"{ylabel:s} + {yshift:.2e}"
    elif yscale != 1.0:
      ylabel = rf"{ylabel:s} $\times$ {yscale:.2e}"
    # end
  # end
  if zscale != 1.0:
    if clabel:
      clabel = rf"{clabel:s} $\times$ {zscale:.3e}"
    else:
      clabel = rf"$\times$ {zscale:.3e}"
    # end
  # end

  if bool(figsize):
    figsize = (int(figsize.split(",")[0]), int(figsize.split(",")[1]))
  # end
  if squeeze or num_comps == 1:
    fig = go.Figure()
    scene_names = ["scene"]
    grid_shape = (1, 1)
  else:
    if num_subplot_row is not None:
      num_rows = num_subplot_row
      num_cols = int(np.ceil(num_comps / num_rows))
    elif num_subplot_col is not None:
      num_cols = num_subplot_col
      num_rows = int(np.ceil(num_comps / num_cols))
    else:
      sr = np.sqrt(num_comps)
      if sr == np.ceil(sr):
        num_rows = int(sr)
        num_cols = int(sr)
      elif np.ceil(sr) * np.floor(sr) >= num_comps:
        num_rows = int(np.floor(sr))
        num_cols = int(np.ceil(sr))
      else:
        num_rows = int(np.ceil(sr))
        num_cols = int(np.ceil(sr))
      # end
    # end
    specs = [[{"type": "scene"} for _ in range(num_cols)] for _ in range(num_rows)]
    fig = make_subplots(rows=num_rows, cols=num_cols, specs=specs)
    scene_names = ["scene" if idx == 0 else f"scene{idx + 1}" for idx in range(num_comps)]
    grid_shape = (num_rows, num_cols)
  # end

  colorscale = _plotly_colorscale(mpl.rcParams["image.cmap"])
  scalar_colorscale = [[0.0, color], [1.0, color]] if bool(color) else colorscale
  paper_color = theme_colors["paper_color"]
  scene_color = theme_colors["scene_color"]
  text_color = theme_colors["text_color"]
  grid_color = theme_colors["grid_color"]
  axis_line_color = theme_colors["axis_line_color"]

  fig.update_layout(
    paper_bgcolor=paper_color,
    plot_bgcolor=paper_color,
    font=dict(color=text_color),
  )

  slice_planes: list[tuple[int, int | float, list[np.ndarray], np.ndarray]] = []
  if slice_plane:
    if isinstance(data, tuple):
      raise ValueError("slice_plane rendering requires GData input")
    # end
    for axis_key in ("z0", "z1", "z2"):
      if axis_key not in slice_plane:
        continue
      # end
      slice_axis = int(axis_key[1:])
      axis_values = slice_plane[axis_key]
      if isinstance(axis_values, (list, tuple, np.ndarray)):
        selector_values = list(axis_values)
      else:
        selector_values = [axis_values]
      # end
      for axis_value in selector_values:
        slice_grid, slice_values = data_select(data, **{axis_key: axis_value})
        slice_planes.append((slice_axis, axis_value, slice_grid, slice_values))
      # end
    # end
    if not slice_planes:
      raise ValueError("3D slicing only supports z0, z1, or z2")
    # end
  # end

  colorbar_kwargs = dict(
    title=dict(text=clabel or "", font=dict(color=text_color)),
    exponentformat="e",
    showexponent="all",
    tickfont=dict(color=text_color),
    bgcolor=paper_color,
  )

  for comp_idx, comp in enumerate(idx_comps):
    if comp_idx >= len(scene_names):
      break
    # end
    scene_name = scene_names[comp_idx]
    row = 1 if grid_shape == (1, 1) else int(comp_idx / grid_shape[1]) + 1
    col = 1 if grid_shape == (1, 1) else int(comp_idx % grid_shape[1]) + 1
    label = f"{label_prefix:s}_c{comp:d}".strip("_") if len(idx_comps) > 1 else label_prefix
    nodal_grid = _get_nodal_grid(grid, cells)
    value = np.asarray(values[..., comp]) * zscale + zshift
    color_value = value * cscale + cshift
    x_grid, y_grid, z_grid = _prepare_3d_coordinates(nodal_grid, value.shape)
    x_coord = np.asarray(x_grid)
    y_coord = np.asarray(y_grid)
    if cylindrical_to_cartesian:
      r = x_coord
      theta = y_coord
      x_coord = r * np.cos(theta)
      y_coord = r * np.sin(theta)
    # end
    x = (x_coord + xshift) * xscale
    y = (y_coord + yshift) * yscale
    z = np.asarray(z_grid)
    finite_value = np.isfinite(color_value)
    finite_count = int(finite_value.sum())
    if finite_count:
      value_min = float(np.nanmin(color_value))
      value_max = float(np.nanmax(color_value))
    else:
      value_min = float("nan")
      value_max = float("nan")
    # end

    z_axis_label = _latex_to_html(zlabel) if zlabel else _latex_to_html(axes_labels[2])
    x_axis_range = _axis_range(x, xrange, logx)
    y_axis_range = _axis_range(y, yrange, logy)
    z_axis_range = _axis_range(z, zrange, logz)
    scene_aspectmode, scene_aspectratio = _resolve_plotly_aspect(aspect, fixaspect)

    scene = dict(
      xaxis=dict(
        title=dict(text=_latex_to_html(xlabel), font=dict(color=text_color)), showgrid=showgrid,
        type="log" if logx else "linear", exponentformat="e", range=x_axis_range,
        showbackground=True, backgroundcolor=scene_color, gridcolor=grid_color,
        linecolor=axis_line_color, tickfont=dict(color=text_color),
        zerolinecolor=grid_color,
      ),
      yaxis=dict(
        title=dict(text=_latex_to_html(ylabel), font=dict(color=text_color)), showgrid=showgrid,
        type="log" if logy else "linear", exponentformat="e", range=y_axis_range,
        showbackground=True, backgroundcolor=scene_color, gridcolor=grid_color,
        linecolor=axis_line_color, tickfont=dict(color=text_color),
        zerolinecolor=grid_color,
      ),
      zaxis=dict(
        title=dict(text=z_axis_label, font=dict(color=text_color)), showgrid=showgrid,
        type="log" if logz else "linear", exponentformat="e", range=z_axis_range,
        showbackground=True, backgroundcolor=scene_color, gridcolor=grid_color,
        linecolor=axis_line_color, tickfont=dict(color=text_color),
        zerolinecolor=grid_color,
      ),
      bgcolor=scene_color,
      aspectmode=scene_aspectmode,
      aspectratio=scene_aspectratio,
    )
    fig.update_layout(**{scene_name: scene})

    # Determine color range (same for both slice and volume rendering)
    if diverging:
      cmax_val = float(np.nanmax(np.abs(color_value)))
      cmin_val = -cmax_val
    else:
      if clim is not None:
        cmin_local, cmax_local = clim
      else:
        cmin_local = cmin if cmin is not None else None
        cmax_local = cmax if cmax is not None else None
      # end
      cmin_val = cmin_local if cmin_local is not None else value_min
      cmax_val = cmax_local if cmax_local is not None else value_max
    # end

    trace_colorscale = scalar_colorscale
    trace_colorbar_kwargs = dict(colorbar_kwargs)

    if slice_planes:
      render_color_value = np.array(color_value, copy=True)
      render_x, render_y, render_z = x, y, z
      volume_opacity_scale = [[0.0, 0.0], [0.5, 0.2], [1.0, 0.75]]
      show_volume_colorbar = False
    else:
      render_color_value = np.array(color_value, copy=True)
      if logz:
        positive = np.where(render_color_value > 0, render_color_value, np.nan)
        render_color_value = np.log10(positive)
        if cmin_val is not None:
          cmin_val = np.log10(max(cmin_val, np.finfo(float).tiny))
        # end
        if cmax_val is not None:
          cmax_val = np.log10(cmax_val)
        # end
      # end
      render_x, render_y, render_z = x, y, z
      volume_opacity_scale = [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8]]
      show_volume_colorbar = colorbar and comp_idx == 0 and not bool(color)
    # end

    if logc:
      log_value = np.full(render_color_value.shape, np.nan, dtype=float)
      valid_mask = render_color_value > 0
      log_value[valid_mask] = np.log10(render_color_value[valid_mask])

      if np.any(valid_mask):
        valid_min = float(np.nanmin(log_value[valid_mask]))
        valid_max = float(np.nanmax(log_value[valid_mask]))
      else:
        valid_min = 0.0
        valid_max = 1.0
      # end

      if cmin_val is not None and cmin_val > 0:
        valid_min = float(np.log10(cmin_val))
      # end
      if cmax_val is not None and cmax_val > 0:
        valid_max = float(np.log10(cmax_val))
      # end
      if not np.isfinite(valid_max) or valid_max <= valid_min:
        valid_max = valid_min + 1.0
      # end

      render_color_value = np.nan_to_num(log_value, nan=valid_min, posinf=valid_max, neginf=valid_min)
      cmin_val = valid_min
      cmax_val = valid_max

      tick_vals, tick_text = _log_colorbar_ticks(cmin_val, cmax_val)
      if tick_vals:
        trace_colorbar_kwargs["tickmode"] = "array"
        trace_colorbar_kwargs["tickvals"] = tick_vals
        trace_colorbar_kwargs["ticktext"] = tick_text
      # end
    # end

    if slice_planes:
      xv, yv, zv, render_color_value = _downsample_3d_volume(
          render_x, render_y, render_z, render_color_value,
          maximum_points_per_axis=maximum_points_per_axis,
      )

      volume_trace = go.Volume(
          x=xv.ravel(), y=yv.ravel(), z=zv.ravel(), value=render_color_value.ravel(),
          colorscale=trace_colorscale,
          cmin=cmin_val,
          cmax=cmax_val,
          opacity=opacity,
          opacityscale=volume_opacity_scale,
          surface_count=surface_count,
          showscale=False,
          name=(label or f"c{comp}") + "_volume",
          showlegend=False,
      )
      trace_list = [volume_trace]

      for plane_idx, (slice_axis, slice_selector, slice_grid, slice_values) in enumerate(slice_planes):
        slice_value = np.squeeze(np.asarray(slice_values[..., comp])) * zscale + zshift
        slice_color_value = slice_value * cscale + cshift

        if logc:
          log_slice = np.full(slice_color_value.shape, np.nan, dtype=float)
          valid_mask = slice_color_value > 0
          log_slice[valid_mask] = np.log10(slice_color_value[valid_mask])
          slice_color_value = np.nan_to_num(
              log_slice,
              nan=cmin_val,
              posinf=cmax_val,
              neginf=cmin_val,
          )
        # end

        plane_index = _resolve_slice_plane_index(grid[slice_axis], slice_selector, value.shape[slice_axis])
        if slice_axis == 0:
          sx = x[plane_index, :, :]
          sy = y[plane_index, :, :]
          sz = z[plane_index, :, :]
        elif slice_axis == 1:
          sx = x[:, plane_index, :]
          sy = y[:, plane_index, :]
          sz = z[:, plane_index, :]
        else:
          sx = x[:, :, plane_index]
          sy = y[:, :, plane_index]
          sz = z[:, :, plane_index]
        # end
        sc = np.asarray(slice_color_value)

        surface_trace = go.Surface(
            x=sx,
            y=sy,
            z=sz,
            surfacecolor=sc,
            colorscale=scalar_colorscale,
          cmin=cmin_val,
          cmax=cmax_val,
            showscale=colorbar and comp_idx == 0 and not bool(color) and plane_idx == 0,
          colorbar=trace_colorbar_kwargs if colorbar and comp_idx == 0 and not bool(color) and plane_idx == 0 else None,
            opacity=opacity,
            name=(label or f"c{comp}") + f"_slice{plane_idx}",
            showlegend=legend and bool(label) and plane_idx == 0,
        )
        trace_list.append(surface_trace)
      # end
    else:
      render_x, render_y, render_z, render_color_value = _downsample_3d_volume(
          render_x, render_y, render_z, render_color_value,
          maximum_points_per_axis=maximum_points_per_axis,
      )
      trace = go.Volume(
          x=render_x.ravel(), y=render_y.ravel(), z=render_z.ravel(), value=render_color_value.ravel(),
          colorscale=trace_colorscale,
          cmin=cmin_val,
          cmax=cmax_val,
          opacity=opacity,
          opacityscale=volume_opacity_scale,
          surface_count=surface_count,
          showscale=show_volume_colorbar,
          colorbar=trace_colorbar_kwargs if show_volume_colorbar else None,
          name=label or f"c{comp}",
          showlegend=legend and bool(label),
      )
      trace_list = [trace]
    # end

    for trace in trace_list:
      if grid_shape == (1, 1):
        fig.add_trace(trace)
      else:
        fig.add_trace(trace, row=row, col=col)
    # end
  # end

  if bool(title):
    fig.update_layout(title=title)
  # end
  if bool(hashtag):
    fig.add_annotation(text="#pgkyl", x=0.99, y=0.01, xref="paper", yref="paper",
        showarrow=False, xanchor="right", yanchor="bottom")
  # end
  if bool(figsize):
    fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)
  # end
  fig.update_layout(margin=dict(l=10, r=10, t=40 if title else 10, b=10))
  return fig


__all__ = ["plot3d", "save_rotating_plotly_figure"]
