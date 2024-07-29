import numpy as np


def rel_change(dataset0, dataset, comp=None):
  """Function to compute the relative change in a dataset compared to another
  dataset, i.e. (dataset - dataset0)/dataset0

  Notes:
  Assumes user wishes to perform this operation component-wise.
  Also assumes the reference division should be performed with respect to a single
  component (i.e., for energetics, divide by the total energy,
  not an individual component of the energy)
  """
  # Grid is the same for each of the input objects
  grid = dataset.get_grid()
  values = dataset.get_values()
  values0 = dataset0.get_values()
  out = np.zeros(values.shape)
  for i in range(0, out.shape[-1]):
    if comp is not None:
      out[..., i] = (values[..., i] - values0[..., i]) / values0[..., int(comp)]
    else:
      out[..., i] = (values[..., i] - values0[..., i]) / values0[..., i]
  return grid, out
