import numpy as np

from postgkyl.data import GData

def info(data: GData) -> str:
    """Prints GData object information.

    Prints time (only when available), number of components, dimension
    spans, extremes for a GData object. Only the top of the stack is
    printed.

    Args:
        data (GData): A GData object to be printed

    Returns:
        output (str): A list of strings with the informations
    """
    grid, values = data.peakStack()
    numComps = data.getNumComps()
    numDims = data.getNumDims()
    numCells = data.getNumCells()
    lower, upper = data.getBounds()

    maximum = values.max()
    maxIdx = np.unravel_index(np.argmax(values), values.shape)
    minimum = values.min()
    minIdx = np.unravel_index(np.argmin(values), values.shape)

    output = ""
    if data.time is not None:
        output += "- Time: {:e}\n".format(data.time)
    output += "- Number of components: {:d}\n".format(numComps)
    output += "- Number of dimensions: {:d}\n".format(numDims)
    for d in range(numDims):
        output += "  - Dim {:d}: Num. cells: {:d}; Lower: {:e}; Upper: {:e}\n".format(d, numCells[d], lower[d], upper[d])
    output += "- Maximum: {:e} at {:s}\n".format(maximum, str(maxIdx))
    output += "- Minimum: {:e} at {:s}".format(minimum, str(minIdx))
    return output
