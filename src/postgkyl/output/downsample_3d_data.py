import numpy as np

def downsample_3d_data(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    value: np.ndarray,
    maximum_points_per_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Downsample 3D arrays so no axis exceeds the configured maximum.
  Axes and values must have the same shape. (i.e. both must be cell centered or both must be nodal.)
  """
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