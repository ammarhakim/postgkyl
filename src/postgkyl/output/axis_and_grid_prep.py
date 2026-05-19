import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
  from postgkyl import GData

def _default_axis_labels(num_dims: int) -> list[str]:
  """Return default axis labels matching plot.py style."""
  return [rf"$z_{i}$" for i in range(num_dims)]

def _format_axis_label(label: str, shift: float, scale: float) -> str:
  """Format axis labels with shift/scale annotation, matching plot.py behavior."""
  if shift != 0.0 and scale != 1.0:
    return rf"({label:s} + {shift:.2e}) $\times$ {scale:.2e}"
  if shift != 0.0:
    return rf"{label:s} + {shift:.2e}"
  if scale != 1.0:
    return rf"{label:s} $\times$ {scale:.2e}"
  return label


def _resolve_plot_labels(
    xlabel: str | None,
    ylabel: str | None,
    zlabel: str | None,
    clabel: str,
    xshift: float,
    yshift: float,
    zshift: float,
    xscale: float,
    yscale: float,
    zscale: float,
    num_dims: int,
) -> tuple[str, str, str, str]:
  """Infer defaults and apply formatting to axis/colorbar labels."""
  axis_labels = _default_axis_labels(num_dims)

  if xlabel is None:
    xlabel = axis_labels[0]
  if ylabel is None:
    ylabel = axis_labels[1] if num_dims > 1 else axis_labels[0]
  if zlabel is None:
    zlabel = axis_labels[2] if num_dims > 2 else axis_labels[-1]

  xlabel = _format_axis_label(xlabel, xshift, xscale)
  ylabel = _format_axis_label(ylabel, yshift, yscale)
  zlabel = _format_axis_label(zlabel, zshift, zscale)

  if zscale != 1.0:
    if clabel:
      clabel = rf"{clabel:s} $\times$ {zscale:.3e}"
    else:
      clabel = rf"$\times$ {zscale:.3e}"

  return xlabel, ylabel, zlabel, clabel


def axis_and_grid_prep(
    grid: list[np.ndarray],
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cells: np.ndarray,
    num_dims: int,
    streamline: bool,
    quiver: bool,
    num_axes: int | None,
    lineouts: int | None,
    xlabel: str | None,
    ylabel: str | None,
    zlabel: str | None,
    clabel: str | None,
    xshift: float,
    yshift: float,
    zshift: float,
    xscale: float,
    yscale: float,
    zscale: float,
) -> tuple[
    list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
  int, range, str, str | None, str | None, str,
]:
  """Apply plot.py preprocessing for collapsed dims, components, and labels."""
  axes_labels = np.array(_default_axis_labels(max(6, len(grid))), dtype=object)

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

  step = 2 if bool(streamline or quiver) else 1
  num_comps = values.shape[-1]
  idx_comps = range(int(np.floor(num_comps / step)))
  if num_axes:
    num_comps = num_axes
  else:
    num_comps = len(idx_comps)
  # end

  if xlabel is None:
    xlabel = axes_labels[0] if lineouts != 1 else axes_labels[1]
  # end
  if ylabel is None and num_dims == 2 and lineouts is None:
    ylabel = axes_labels[1]
  # end
  xlabel, ylabel, zlabel, clabel = _resolve_plot_labels(
    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
    clabel=clabel,
    xshift=xshift, yshift=yshift, zshift=zshift,
    xscale=xscale, yscale=yscale, zscale=zscale,
    num_dims=num_dims,
  )

  return grid, values, lower, upper, cells, axes_labels, num_comps, idx_comps, xlabel, ylabel, zlabel, clabel