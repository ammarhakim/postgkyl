import numpy as np
from scipy import fftpack

class Fft(object):
    """Postgkyl class for the Fast Fourier Transformation.

    Init Args:
        data (GData): data set
    """

    def __init__(self, data):
        self.data = data

    def fft(self, psd=False, stack=False):
        grid = self.data.peakGrid()
        values = self.data.peakValues() 

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
        
        numComps = self.data.getNumComps()
        if numDims > 1:
            raise ValueError("Only 1D darta are currently supported.")

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
            self.data.pushGrid(freq, lo, up)
            self.data.pushValues(ftValues)
        else:
            return freq, ftValues
        
