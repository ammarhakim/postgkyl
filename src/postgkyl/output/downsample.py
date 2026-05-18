import numpy as np


def downsample(
    *arrays: np.ndarray,
    maximum_points_per_axis: int = 0,
) -> tuple[np.ndarray, ...]:
  """Downsample same-shape arrays so no axis exceeds the configured maximum.

  This is dimension-agnostic and works for any array dimensionality.

  Args:
    *arrays: One or more arrays to downsample. All arrays must have the same shape.
    maximum_points_per_axis: The maximum number of points allowed along any axis after downsampling. If 0 or negative, no downsampling is performed.
  Returns:
    A tuple of downsampled arrays corresponding to the input arrays.

  Example:
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    z = np.linspace(0, 10, 100)
    value = np.random.rand(100, 100, 100)
    x_ds, y_ds, z_ds, value_ds = downsample_data(x, y, z, value, maximum_points_per_axis=20)

  """
  if not arrays:
    return ()
  # end

  reference = arrays[0]
  if maximum_points_per_axis is None or maximum_points_per_axis <= 0:
    return arrays
  # end

  if reference.ndim == 0:
    return arrays
  # end

  if any(arr.shape != reference.shape for arr in arrays):
    return arrays
  # end

  steps = [
      max(1, int(np.ceil(size / maximum_points_per_axis)))
      for size in reference.shape
  ]
  if max(steps) == 1:
    return arrays
  # end

  def _axis_indices(size: int, step: int) -> np.ndarray:
    idx = np.arange(0, size, step, dtype=int)
    if idx[-1] != size - 1:
      idx = np.append(idx, size - 1)
    # end
    return idx

  axis_indices = [
      _axis_indices(size, step)
      for size, step in zip(reference.shape, steps)
  ]

  def _take_indices(arr: np.ndarray) -> np.ndarray:
    out = arr
    for axis, idx in enumerate(axis_indices):
      out = np.take(out, idx, axis=axis)
    # end
    return out

  return tuple(_take_indices(arr) for arr in arrays)