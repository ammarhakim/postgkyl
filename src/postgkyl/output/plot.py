"""Module including custom Gkeyll plotting function"""
from __future__ import annotations

from itertools import product
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, TYPE_CHECKING
import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import os.path

try:
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - optional dependency
  go = None
  make_subplots = None

from postgkyl.utils import input_parser
if TYPE_CHECKING:
  from postgkyl import GData
# end

# Helper functions
def pgkyl_colorbar(obj, fig : matplotlib.figure.Figure, cax : matplotlib.axes.Axes,
    label: str = "", extend: bool | None = None):
  divider = make_axes_locatable(cax)
  cax2 = divider.append_axes("right", size="3%", pad=0.05)
  return fig.colorbar(obj, cax=cax2, label=label or "", extend=extend)


def _apply_plot_style(style: str | None, rcParams: dict | None, diverging: bool,
    cmap: str | None, jet: bool, xkcd: bool) -> None:
  if bool(style):
    plt.style.use(style)
  elif bool(rcParams):
    for key in rcParams:
      mpl.rcParams[key] = rcParams[key]
    # end
  else:
    plt.style.use(f"{os.path.dirname(os.path.realpath(__file__)):s}/postgkyl.mplstyle")
  # end

  if bool(cmap):
    mpl.rcParams["image.cmap"] = cmap
  elif bool(diverging):
    mpl.rcParams["image.cmap"] = "RdBu_r"
  # end

  if bool(jet):
    mpl.rcParams["image.cmap"] = "jet"
  # end

  if xkcd:
    plt.xkcd()
  # end


def _plotly_colorscale(cmap_name: str, n: int = 256):
  cmap = mpl.colormaps.get_cmap(cmap_name).resampled(n)
  xs = np.linspace(0.0, 1.0, n)
  colorscale = []
  for x, rgba in zip(xs, cmap(xs)):
    r, g, b, a = rgba
    colorscale.append([float(x), f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {float(a):.3f})"])
  # end
  return colorscale


def _plot_debug(message: str) -> None:
  print(f"[postgkyl.plot] {message}")
  # end


def _finite_range(values: np.ndarray) -> tuple[float, float]:
  finite = np.isfinite(values)
  if np.any(finite):
    finite_values = values[finite]
    return float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
  # end
  return float("nan"), float("nan")


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


def _infer_num_dims(data: GData | Tuple[list, np.ndarray]) -> int:
  grid, values = input_parser(data)
  if isinstance(data, tuple):
    if len(grid) == len(values.shape):
      return len(values.squeeze().shape)
    else:
      return len(values[..., 0].squeeze().shape)
    # end
  else:
    return data.get_num_dims(squeeze=True)
  # end


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


def plot_matplotlib(data: GData | Tuple[list, np.ndarray], args: list = (),
    figure: int | matplotlib.figure.Figure | str | None = None,
    squeeze: bool = False, num_axes: int = None, start_axes: int = 0,
    num_subplot_row: int | None = None, num_subplot_col: int | None = None,
    streamline: bool = False, sdensity: int = 1,
    quiver: bool = False,
    contour: bool = False, clevels: list | None = None, cnlevels: int | None = None, cont_label: bool = False,
    diverging: bool = False,
    lineouts: int | None = None,
    xmin: float | None = None, xmax: float | None = None, xscale: float = 1.0, xshift: float = 0.0,
    ymin: float | None = None, ymax: float | None = None, yscale: float = 1.0, yshift: float = 0.0,
    zmin: float | None = None, zmax: float | None = None, zscale: float = 1.0, zshift: float = 0.0,
    relax: bool = False, style: str | None = None, rcParams: dict | None = None,
    legend: bool = True, label_prefix: str = "", colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None, clabel: str | None = None, title: str | None = None,
    subplot_titles: str | None = None, subplot_xlabels: str | None = None, subplot_ylabels: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False, logc: bool = False,
    fixaspect: bool = False, aspect: float | None = None,
    edgecolors: str | None = None, showgrid: bool = True, hashtag: bool = False, xkcd: bool = False,
    color: str | None = None, markersize: float | None = None,
    linewidth: float | None = None, linestyle: float | None = None, opacity: float | None = None,
    figsize: tuple | None = None,
    jet: bool = False, cmap: str | None = None,
    **kwargs):
  """Plots Gkeyll data.

  Unifies the plotting across a wide range of Gkyl applications. Can
  be used for both 1D an 2D data. Uses a proper colormap by default.
  """

  # ---- Set style and process inputs ----
  # Default to Postgkyl style file file if no style is specified
  # Use the rcParams dictionary which is passed with click contex
  _apply_plot_style(style, rcParams, diverging, cmap, jet, xkcd)

  # Process input parameters
  if not bool(aspect):
    aspect = 1.0
  # end

  if not bool(color) and not isinstance(data, tuple):
    cl = data.color
  # end
  if bool(color):
    mpl.rcParams["lines.color"] = color
  # end
  if bool(linewidth):
    mpl.rcParams["lines.linewidth"] = linewidth
  # end
  if bool(linestyle):
    mpl.rcParams["lines.linestyle"] = linestyle
  # end

  # ---- Data Loading ----
  # Get the handles on the grid and values
  grid_in, values = input_parser(data)
  grid = grid_in.copy()

  if isinstance(data, tuple):
    if len(grid) == len(values.shape):
      num_dims = len(values.squeeze().shape)
    else:
      num_dims = len(values[..., 0].squeeze().shape)
    # end
    lg = len(grid)
    lower, upper, cells = np.zeros(lg), np.zeros(lg), np.zeros(lg)
    for d in range(lg):
      lower[d] = np.min(grid[d])
      upper[d] = np.max(grid[d])
      if len(grid[d].shape) == 1:
        cells[d] = len(grid[d])
      else:
        cells[d] = len(grid[d][d])
      # end
    # end
  else: # GData
    num_dims = data.get_num_dims(squeeze=True)
    lower, upper = data.get_bounds()
    cells = data.get_num_cells()
  # end
  if num_dims > 2:
    raise ValueError("Only 1D and 2D plots are currently supported")
  # end

  # Squeeze the data (get rid of "collapsed" dimensions)
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
      lower = np.delete(lower, idx)
      upper = np.delete(upper, idx)
      cells = np.delete(cells, idx)
      axes_labels = np.delete(axes_labels, idx)
      values = np.squeeze(values, tuple(idx))

      # c2p grids
      if len(grid[0].shape) > 1:
        for d in range(num_dims):
          for i in reversed(idx):
            grid[d] = np.mean(grid[d], axis=i)
          # end
        # end
      # end
    # end
  # end

  # Get the number of components and an indexer
  step = 2 if bool(streamline or quiver) else 1
  num_comps = values.shape[-1]
  idx_comps = range(int(np.floor(num_comps / step)))
  if num_axes:
    num_comps = num_axes
  else:
    num_comps = len(idx_comps)
  # end

  # Create axis labels
  if xlabel is None:
    xlabel = axes_labels[0] if lineouts != 1 else axes_labels[1]
    if xshift != 0.0 and xscale != 1.0:
      xlabel = rf"({xlabel:s} + {xshift:.2e}) $\times$ {xscale:.2e}"
    elif xshift != 0.0:
      xlabel = rf"{xlabel:s} + {xshift:.2e}"
    elif xscale != 1.0:
      xlabel = rf"{xlabel:s} $\times$ {xscale:.2e}"
    # end
  # end
  if ylabel is None and num_dims == 2 and lineouts is None:
    ylabel = axes_labels[1]
    if yshift != 0.0 and yscale != 1.0:
      ylabel = rf"({ylabel:s} + {yshift:.2e}) $\times$ {yscale:.2e}"
    elif xshift != 0.0:
      ylabel = rf"{ylabel:s} + {yshift:.2e}"
    elif xscale != 1.0:
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

  # ---- Prepare Figure and Axes ----------------------------------------
  if bool(figsize):
    figsize = (int(figsize.split(",")[0]), int(figsize.split(",")[1]))
  # end
  if figure is None:
    fig = plt.figure(figsize=figsize)
  elif isinstance(figure, int):
    fig = plt.figure(figure, figsize=figsize)
  elif isinstance(figure, matplotlib.figure.Figure):
    fig = figure
  elif isinstance(figure, str):
    fig = plt.figure(int(figure), figsize=figsize)
  else:
    raise TypeError(
        "'fig' keyword needs to be one of " "None (default), int, or MPL Figure"
    )
  # end

  # Axes
  if fig.axes:
    ax = fig.axes
    if squeeze is False and num_comps > len(ax):
      raise ValueError("Trying to plot into figure with not enough axes")
    # end
  else:
    if squeeze:  # Plotting into 1 panel
      plt.subplots(1, 1, num=fig.number)
      ax = fig.axes
      ax[0].set_xlabel(xlabel)
      ax[0].set_ylabel(ylabel)
      if title is not None:
        ax[0].set_title(title, y=1.08)
      # end
    else:  # Plotting each components into its own subplot
      if num_subplot_row is not None:
        num_rows = num_subplot_row
        num_cols = int(np.ceil(num_comps/num_rows))
      elif num_subplot_col is not None:
        num_cols = num_subplot_col
        num_rows = int(np.ceil(num_comps/num_cols))
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

      if num_dims == 1 or lineouts is not None:
        plt.subplots(num_rows, num_cols, sharex=True, num=fig.number)
      else:  # In 2D, share y-axis as well
        plt.subplots(num_rows, num_cols, sharex=True, sharey=True, num=fig.number)
      # end
      ax = fig.axes
      # Removing extra axes
      for i in range(num_comps, len(ax)):
        ax[i].axis("off")
      # end
      # Add labels as super labels and titles
      if bool(title):
        fig.suptitle(title)
      if bool(xlabel):
        fig.supxlabel(xlabel)
      if bool(ylabel):
        fig.supylabel(ylabel)

      for ax_idx, _ in enumerate(ax):
        if bool(subplot_titles):
          title = subplot_titles.split(",")[ax_idx] if ax_idx < len(subplot_titles.split(",")) else ""
        else:
          title = ""
        # end
        if bool(subplot_xlabels):
          xlabel = subplot_xlabels.split(",")[ax_idx] if ax_idx < len(subplot_xlabels.split(",")) else ""
        else:
          xlabel = ""
        # end
        if bool(subplot_ylabels):
          ylabel = subplot_ylabels.split(",")[ax_idx] if ax_idx < len(subplot_ylabels.split(",")) else ""
        else:
          ylabel = ""
        # end

        ax[ax_idx].set_xlabel(xlabel)
        ax[ax_idx].set_ylabel(ylabel)
        if bool(title):
          ax[ax_idx].set_title(title, y=1.08)
        # end
      # end
    # end
  # end

  # ---- Main Plotting Loop ---------------------------------------------
  for comp in idx_comps:
    cax = ax[0] if squeeze else ax[comp + start_axes]
    label = f"{label_prefix:s}_c{comp:d}".strip("_") if len(idx_comps) > 1 else label_prefix

    if num_dims == 1:
      nodal_grid = _get_nodal_grid(grid, cells)
      x = (nodal_grid[0] + xshift)*xscale
      y = (values[..., comp] + yshift)*yscale
      im = cax.plot(x, y, *args, color=color, label=label, markersize=markersize)

    elif num_dims == 2:
      extend = None

      if contour:  # ----------------------------------------------------
        levels = 10
        if cnlevels:
          levels = int(cnlevels) - 1
        elif clevels:
          if ":" in clevels:
            s = clevels.split(":")
            levels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
          else:
            levels = np.array(clevels.split(","))
            # Filter out empty elements
            levels = np.array(list(filter(None, levels)))
          # end
        # end
        if isinstance(levels, np.ndarray) and len(levels) == 1:
          colorbar = False
        # end
        nodal_grid = _get_nodal_grid(grid, cells)
        x = (nodal_grid[0] + xshift) * xscale
        y = (nodal_grid[1] + yshift) * yscale
        z = (values[..., comp].transpose() + zshift) * zscale
        im = cax.contour(x, y, z, levels, *args, origin="lower", colors=color, linewidths=linewidth)
        if cont_label:
          cax.clabel(im, inline=1)
        # end

      elif quiver:  # ----------------------------------------------------
        skip = int(np.max((len(grid[0]), len(grid[1])))//15)
        skip2 = int(skip//2)
        nodal_grid = _get_nodal_grid(grid, cells)
        if len(nodal_grid[0].shape) == 1:
          x = (nodal_grid[0][skip2::skip] + xshift)*xscale
          y = (nodal_grid[1][skip2::skip] + yshift)*yscale
        else:
          x = (nodal_grid[0][skip2::skip, skip2::skip] + xshift)*xscale
          y = (nodal_grid[1][skip2::skip, skip2::skip] + yshift)*yscale
        # end
        z1 = (values[skip2::skip, skip2::skip, 2 * comp].transpose() + zshift)*zscale
        z2 = (values[skip2::skip, skip2::skip, 2 * comp + 1].transpose() + zshift)*zscale
        im = cax.quiver(x, y, z1, z2)

      elif streamline:  # ------------------------------------------------
        if bool(color):
          cl = color
        else:
          # magnitude
          cl = np.sqrt(
              values[..., 2 * comp]**2 + values[..., 2 * comp + 1]**2
          ).transpose()
        # end
        nodal_grid = _get_nodal_grid(grid, cells)
        x = (nodal_grid[0] + xshift)*xscale
        y = (nodal_grid[1] + yshift)*yscale
        z1 = (values[..., 2 * comp].transpose() + zshift)*zscale
        z2 = (values[..., 2 * comp + 1].transpose() + zshift)*zscale
        im = cax.streamplot(x, y, z1, z2, *args,
            density=sdensity, broken_streamlines=False, color=cl, linewidth=linewidth)

      elif lineouts is not None:  # -------------------------------------
        num_lines = values.shape[1] if lineouts == 0 else values.shape[0]
        nodal_grid = _get_nodal_grid(grid, cells)

        if lineouts == 0:
          x = (nodal_grid[0] + xshift)*xscale
          vmin = (nodal_grid[1][0] + yshift)*yscale
          vmax = (nodal_grid[1][-1] + yshift)*yscale
          label = clabel or axes_labels[1]
        else:
          x = (nodal_grid[1] + xshift)*xscale
          vmin = (nodal_grid[0][0] + yshift)*yscale
          vmax = (nodal_grid[0][-1] + yshift)*yscale
          label = clabel or axes_labels[0]
        # end
        idx = [slice(0, u) for u in values.shape]
        idx[-1] = comp
        for line in range(num_lines):
          color = cm.inferno(line / (num_lines - 1))
          if lineouts == 0:
            idx[1] = line
          else:
            idx[0] = line
          # end
          y = (values[tuple(idx)] + yshift)*yscale
          im = cax.plot(x, y, *args, color=color)
        # end
        mappable = cm.ScalarMappable(
            norm=colors.Normalize(vmin=vmin, vmax=vmax, clip=False), cmap=cm.inferno
        )
        pgkyl_colorbar(mappable, fig, cax, label=label)
        colorbar = False
        legend = False

      else:  # -----------------------------------------------------------
        if zmin is not None and zmax is not None:
          extend = "both"
        elif zmax is not None:
          extend = "max"
        elif zmin is not None:
          extend = "min"
        # end
        x = (grid[0] + xshift)*xscale
        y = (grid[1] + yshift)*yscale
        z = (values[..., comp].transpose() + zshift)*zscale
        if len(x) == z.shape[1] or len(y) == z.shape[0]:
          nodal_grid = _get_nodal_grid(grid, cells)
          x = (nodal_grid[0] + xshift)*xscale
          y = (nodal_grid[1] + yshift)*yscale
        # end
        if len(x.shape) > 1:
          x, y = x.transpose(), y.transpose()
        # end
        if diverging:
          zmax = np.abs(z).max()
          zmin = -zmax
        # end
        vmax, vmin = zmax, zmin
        norm = None
        if logz:
          if diverging:
            tmp = vmax/1000
            norm = colors.SymLogNorm(
                linthresh=tmp, linscale=tmp, vmin=vmin, vmax=vmax, base=10
            )
          else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
          # end
          vmin, vmax = None, None
        # end
        im = cax.pcolormesh(x, y, z,
            norm=norm, vmin=vmin, vmax=vmax, edgecolors=edgecolors,
            linewidth=0.1, shading="auto", *args)
      # end
      if not bool(color) and colorbar and not streamline:
        pgkyl_colorbar(im, fig, cax, extend=extend, label=clabel)
      # end
    else:
      raise ValueError(f"{num_dims:d}D data not supported")
    # end

    # ---- Additional Formatting ----------------------------------------
    cax.grid(showgrid)
    # Legend
    if legend:
      if num_dims == 1 and label != "":
        cax.legend(loc=0)
      else:
        cax.text(0.03, 0.96, label,
            bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8, "boxstyle": "round"},
            verticalalignment="top", horizontalalignment="left", transform=cax.transAxes)
      # end
    # end
    if hashtag:
      cax.text(0.97, 0.03, "#pgkyl",
          bbox={"facecolor": "w", "edgecolor": "w", "alpha": 0.8, "boxstyle": "round"},
          verticalalignment="bottom", horizontalalignment="right", transform=cax.transAxes)
    # end
    if logx:
      cax.set_xscale("log")
    # end
    if logy:
      cax.set_yscale("log")
    # end
    if num_dims == 1 and not relax:  # this causes troubles with contours
      plt.autoscale(enable=True, axis="x", tight=True)
      plt.autoscale(enable=True, axis="y")
    # end
    if xmin is not None or xmax is not None:
      cax.set_xlim(xmin, xmax)
    # end
    if ymin is not None or ymax is not None:
      cax.set_ylim(ymin, ymax)
    # end
    if fixaspect:
      plt.setp(cax, aspect=aspect)
    # end
  # end

  plt.tight_layout()
  return im


def _plot_plotly_3d(data: GData | Tuple[list, np.ndarray], args: list = (),
    figure: int | matplotlib.figure.Figure | str | None = None,
    squeeze: bool = False, num_axes: int = None, start_axes: int = 0,
    num_subplot_row: int | None = None, num_subplot_col: int | None = None,
    streamline: bool = False, sdensity: int = 1,
    quiver: bool = False,
    contour: bool = False, clevels: list | None = None, cnlevels: int | None = None, cont_label: bool = False,
    diverging: bool = False,
    lineouts: int | None = None,
    xmin: float | None = None, xmax: float | None = None, xscale: float = 1.0, xshift: float = 0.0,
    ymin: float | None = None, ymax: float | None = None, yscale: float = 1.0, yshift: float = 0.0,
    zmin: float | None = None, zmax: float | None = None, zscale: float = 1.0, zshift: float = 0.0,
    relax: bool = False, style: str | None = None, rcParams: dict | None = None,
    legend: bool = True, label_prefix: str = "", colorbar: bool = True,
    xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None, clabel: str | None = None, title: str | None = None,
    subplot_titles: str | None = None, subplot_xlabels: str | None = None, subplot_ylabels: str | None = None,
    logx: bool = False, logy: bool = False, logz: bool = False, logc: bool = False,
    fixaspect: bool = False, aspect: float | None = None,
    edgecolors: str | None = None, showgrid: bool = True, hashtag: bool = False, xkcd: bool = False,
    color: str | None = None, markersize: float | None = None,
    linewidth: float | None = None, linestyle: float | None = None, opacity: float | None = None,
    figsize: tuple | None = None,
    jet: bool = False, cmap: str | None = None,
    **kwargs):
  """Plots 3D Gkeyll data using Plotly."""

  if go is None or make_subplots is None:
    raise ImportError("Plotly is required for 3D plots")
  # end

  _apply_plot_style(style, rcParams, diverging, cmap, jet, xkcd)

  if not bool(aspect):
    aspect = 1.0
  # end

  grid_in, values = input_parser(data)
  grid = grid_in.copy()

  if isinstance(data, tuple):
    if len(grid) == len(values.shape):
      num_dims = len(values.squeeze().shape)
    else:
      num_dims = len(values[..., 0].squeeze().shape)
    # end
    lg = len(grid)
    lower, upper, cells = np.zeros(lg), np.zeros(lg), np.zeros(lg)
    for d in range(lg):
      lower[d] = np.min(grid[d])
      upper[d] = np.max(grid[d])
      if len(grid[d].shape) == 1:
        cells[d] = len(grid[d])
      else:
        cells[d] = len(grid[d][d])
      # end
    # end
  else:
    num_dims = data.get_num_dims(squeeze=True)
    lower, upper = data.get_bounds()
    cells = data.get_num_cells()
  # end

  if num_dims != 3:
    raise ValueError("Plotly backend only handles 3D data")
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
      lower = np.delete(lower, idx)
      upper = np.delete(upper, idx)
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

  step = 2 if bool(streamline or quiver) else 1
  num_comps = values.shape[-1]
  idx_comps = range(int(np.floor(num_comps / step)))
  if num_axes:
    num_comps = num_axes
  else:
    num_comps = len(idx_comps)
  # end

  if xlabel is None:
    xlabel = axes_labels[0]
    if xshift != 0.0 and xscale != 1.0:
      xlabel = rf"({xlabel:s} + {xshift:.2e}) $\times$ {xscale:.2e}"
    elif xshift != 0.0:
      xlabel = rf"{xlabel:s} + {xshift:.2e}"
    elif xscale != 1.0:
      xlabel = rf"{xlabel:s} $\times$ {xscale:.2e}"
    # end
  # end
  if ylabel is None:
    ylabel = axes_labels[1]
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
    x_grid, y_grid, z_grid = _prepare_3d_coordinates(nodal_grid, value.shape)
    x = (np.asarray(x_grid) + xshift) * xscale
    y = (np.asarray(y_grid) + yshift) * yscale
    z = np.asarray(z_grid)
    x_min, x_max = _finite_range(x)
    y_min, y_max = _finite_range(y)
    z_min_raw, z_max_raw = _finite_range(z)

    finite_xyz = np.isfinite(x).sum() + np.isfinite(y).sum() + np.isfinite(z).sum()
    finite_value = np.isfinite(value)
    finite_count = int(finite_value.sum())
    if finite_count:
      value_min = float(np.nanmin(value))
      value_max = float(np.nanmax(value))
    else:
      value_min = float("nan")
      value_max = float("nan")
    # end

    z_axis_label = _latex_to_html(zlabel) if zlabel else _latex_to_html(axes_labels[2])
    scene = dict(
        xaxis=dict(title=_latex_to_html(xlabel), showgrid=showgrid, type="log" if logx else "linear", exponentformat="e"),
        yaxis=dict(title=_latex_to_html(ylabel), showgrid=showgrid, type="log" if logy else "linear", exponentformat="e"),
        zaxis=dict(title=z_axis_label, showgrid=showgrid, type="log" if logz else "linear", exponentformat="e"),
        aspectmode="manual" if fixaspect else "auto",
        aspectratio=dict(x=aspect, y=aspect, z=aspect) if fixaspect else None,
    )
    fig.update_layout(**{scene_name: scene})

    if quiver and values.shape[-1] >= 3:
      trace = go.Cone(
          x=x.ravel(), y=y.ravel(), z=z.ravel(),
          u=np.asarray(values[..., 0]).ravel(),
          v=np.asarray(values[..., 1]).ravel(),
          w=np.asarray(values[..., 2]).ravel(),
          colorscale=scalar_colorscale,
          cmin=zmin,
          cmax=zmax,
          showscale=colorbar and comp_idx == 0 and not bool(color),
          colorbar=dict(title=clabel or "") if colorbar and comp_idx == 0 and not bool(color) else None,
          sizemode="scaled",
          sizeref=linewidth or 1.0,
          name=label or f"c{comp}",
          showlegend=legend and bool(label),
      )
    elif streamline and values.shape[-1] >= 3:
      trace = go.Streamtube(
          x=x.ravel(), y=y.ravel(), z=z.ravel(),
          u=np.asarray(values[..., 0]).ravel(),
          v=np.asarray(values[..., 1]).ravel(),
          w=np.asarray(values[..., 2]).ravel(),
          colorscale=scalar_colorscale,
          cmin=zmin,
          cmax=zmax,
          showscale=colorbar and comp_idx == 0 and not bool(color),
          colorbar=dict(title=clabel or "") if colorbar and comp_idx == 0 and not bool(color) else None,
          name=label or f"c{comp}",
          showlegend=legend and bool(label),
      )
    else:
      if diverging:
        zmax_local = np.nanmax(np.abs(value))
        zmin_local = -zmax_local
      else:
        zmin_local = zmin
        zmax_local = zmax
      # end
      if zmin_local is None:
        zmin_local = value_min
      # end
      if zmax_local is None:
        zmax_local = value_max
      # end
      if logz:
        positive = np.where(value > 0, value, np.nan)
        value = np.log10(positive)
        if zmin_local is not None:
          zmin_local = np.log10(max(zmin_local, np.finfo(float).tiny))
        # end
        if zmax_local is not None:
          zmax_local = np.log10(zmax_local)
        # end
      # end
      if logc:
        positive_value = np.where(value > 0, value, np.nan)
        value = np.log10(positive_value)
        if zmin_local is not None and zmin_local > 0:
          zmin_local = np.log10(zmin_local)
        # end
        if zmax_local is not None and zmax_local > 0:
          zmax_local = np.log10(zmax_local)
        # end
      # end
      trace = go.Volume(
          x=x.ravel(), y=y.ravel(), z=z.ravel(), value=value.ravel(),
          colorscale=scalar_colorscale,
          cmin=zmin_local,
          cmax=zmax_local,
          opacity=opacity if opacity is not None else 0.5,
          opacityscale=[[0.0, 0.0], [0.5, 0.2], [1.0, 0.8]],
          showscale=colorbar and comp_idx == 0 and not bool(color),
          colorbar=dict(title=clabel or "") if colorbar and comp_idx == 0 and not bool(color) else None,
          name=label or f"c{comp}",
          showlegend=legend and bool(label),
      )
    # end

    if grid_shape == (1, 1):
      fig.add_trace(trace)
    else:
      fig.add_trace(trace, row=row, col=col)
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


def plot(data: GData | Tuple[list, np.ndarray], args: list = (), **kwargs):
  """Dispatch to the Matplotlib or Plotly 3D plotting backend."""
  if _infer_num_dims(data) == 3:
    return _plot_plotly_3d(data, args, **kwargs)
  # end
  return plot_matplotlib(data, args, **kwargs)
