import numpy as np
from scipy import fftpack

def fft(data, psd=False, overwrite=False):
    grid = data.getGrid()
    values = data.getValues() 

    # Remove dummy dimensions
    numDims = len(grid)
    idx = []
    for d in range(numDims):
        if len(grid[d]) <= 2:
            idx.append(d)
        #end
    #end
    if idx:
        grid = np.delete(grid, idx)
        values = np.squeeze(values, tuple(idx)) 
        numDims = len(grid)
    #end

    numComps = data.getNumComps()
    if numDims > 1:
        raise ValueError("Only 1D data are currently supported.")
    #end

    N = len(grid[0])
    dx = grid[0][1] - grid[0][0]
    freq = [fftpack.fftfreq(N, dx)]
    ftValues = np.zeros(values.shape, 'complex')
    for comp in np.arange(numComps):
        ftValues[..., comp] = fftpack.fft(values[..., comp])
    #end

    if psd:
        freq[0] = freq[0][:N//2]
        ftValues = np.abs(ftValues[:N//2, :])**2
    #end

    if overwrite:
        data.push(freq, ftValues)
    else:
        return freq, ftValues
    #end
#end
        
