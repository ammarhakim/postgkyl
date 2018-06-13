import numpy as np
from scipy import fftpack

def fft(data, psd=False, stack=False):
    grid = data.peakGrid()
    values = data.peakValues() 

    # Remove dummy dimensions
    numDims = len(grid)
    idx = []
    for d in range(numDims):
        if len(grid[d]) <= 1:
            idx.append(d)
    if idx:
        grid = np.delete(grid, idx)
        values = np.squeeze(values, tuple(idx)) 
        numDims = len(grid)

    numComps = data.getNumComps()
    if numDims > 1:
        raise ValueError("Only 1D data are currently supported.")

    N = len(grid[0])
    dx = grid[0][1] - grid[0][0]
    freq = [fftpack.fftfreq(N, dx)]
    ftValues = np.zeros(values.shape, 'complex')
    for comp in np.arange(numComps):
        ftValues[..., comp] = fftpack.fft(values[..., comp])

    if psd:
        freq[0] = freq[0][:N//2]
        ftValues = np.abs(ftValues[:N//2, :])**2

    if stack:
        lo = np.array([freq[0][0]])
        up = np.array([freq[0][-1]])
        data.pushGrid(freq, lo, up)
        data.pushValues(ftValues)
    else:
        return freq, ftValues
        
