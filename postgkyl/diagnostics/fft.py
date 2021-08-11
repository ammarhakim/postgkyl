import numpy as np
from scipy import fftpack
from .. import diagnostics as diag

def fft(data, psd=False, iso=False, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
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
    if numDims == 1:
        N = len(grid[0])
        dx = grid[0][1] - grid[0][0]
        freq = [fftpack.fftfreq(N, dx)]
        ftValues = np.zeros(values.shape, 'complex')
        for comp in np.arange(numComps):
            ftValues[..., comp] = fftpack.fft(values[..., comp])
        #end

        if psd:
            freq[0] = freq[0][:N//2]
            if mag:
                ftValues[...,0] = np.sum(np.abs(ftValues[:N//2, :])**2, axis=-1)
            else:
                ftValues = np.abs(ftValues[:N//2, :])**2
            #end
        #end

        if overwrite:
            data.push(freq, ftValues)
        else:
            return freq, ftValues
        #end
    else:
        N = np.zeros(3, dtype=int)
        dx = np.zeros(3)
        freq = []
        for i in range(0, numDims):
            N[i] = len(grid[i])
            dx[i] = grid[i][1] - grid[i][0]
            freq.append(fftpack.fftfreq(N[i], dx[i]))
        #end
        ftValues = np.zeros(values.shape, 'complex')
        for comp in np.arange(numComps):
            ftValues[..., comp] = fftpack.fftn(values[..., comp])
        #end
        if psd:
            for i in range(0, numDims):            
                freq[i] = freq[i][:N[i]//2]
            if numDims == 2:
                ftValues = np.abs(ftValues[:N[0]//2, :N[1]//2, :])**2
                # If only 2D, append third dummy index for ease of logic
                freq.append(0)
            elif numDims == 3:
                ftValues = np.abs(ftValues[:N[0]//2, :N[1]//2, :N[2]//2, :])**2
            else:
                raise ValueError("Only 1D, 2D, and 3D data are currently supported.")
            #end
            if iso:
                nkpolar = int(np.sqrt(np.sum(N[:]**2)))
                nkx = N[0]//2
                nky = N[1]//2
                nkz = N[2]//2
                kx = freq[0]
                ky = freq[1]
                kz = freq[2]
                akp, nbin, polar_index, akplim = diag.initpolar(nkx, nky, nkz, kx, ky, kz, nkpolar)
                fft_iso = np.zeros((nkpolar, numComps))
                for comp in np.arange(numComps):
                    fft_iso[:,comp] = diag.polar_isotropic(nkpolar, nkx, nky, nkz, polar_index, nbin, ftValues[..., comp], kx, ky, kz)
                #end
                # Return isotropic spectra and 1D isotropic ks
                if overwrite:
                    data.push([akp], fft_iso)
                else:
                    return [akp], fft_iso
                #end        
            #end
        #end

        if overwrite and not iso:
            data.push(freq, ftValues)
        else:
            return freq, ftValues
        #end
    #end
#end
        
