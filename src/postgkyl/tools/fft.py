"""Postgkyl module for wrapping FFT."""

from __future__ import annotations

import numpy as np
import scipy.fft
from typing import Tuple, TYPE_CHECKING

from postgkyl.tools.init_polar import init_polar
from postgkyl.tools.polar_isotropic import polar_isotropic
if TYPE_CHECKING:
  from postgkyl import GData
# end

def fft(data: GData, psd: bool = False, iso: bool = False,
    overwrite: bool = False, stack: bool = False) -> Tuple[np.ndarray, np.ndarray]:
  """Postgkyl wrapper of scipy FFT.

  Args:
    data: GData
    psd: bool
      Flag to calculate the Power Spectral Density
    iso: bool
      Flag to return isotropic spectra

    XXX overwrite and stack need refactoring; see laguerre_compose.py
  """
  if stack:
    overwrite = stack
  # end
  grid = data.get_grid()
  values = data.get_values()

  # Remove dummy dimensions
  num_dims = len(grid)
  idx = []
  for d in range(num_dims):
    if len(grid[d]) <= 2:
      idx.append(d)
    # end
  # end
  if idx:
    grid = np.delete(grid, idx)
    values = np.squeeze(values, tuple(idx))
    num_dims = len(grid)
  # end

  num_comps = data.get_num_comps()
  if num_dims == 1:
    N = len(grid[0])
    dx = grid[0][1] - grid[0][0]
    freq = [scipy.fft.fftfreq(N, dx)]
    ft_values = np.zeros(values.shape, "complex")
    for comp in np.arange(num_comps):
      ft_values[..., comp] = scipy.fft.fft(values[..., comp])
    # end

    if psd:
      freq[0] = freq[0][:N//2]
      ft_values = np.abs(ft_values[:N//2, :])**2
    # end

    if overwrite:
      data.push(freq, ft_values)
    else:
      return freq, ft_values
    # end
  else:
    N = np.zeros(3, dtype=int)
    dx = np.zeros(3)
    freq = []
    for i in range(0, num_dims):
      N[i] = len(grid[i])
      dx[i] = grid[i][1] - grid[i][0]
      freq.append(scipy.fft.fftfreq(N[i], dx[i]))
    # end
    ft_values = np.zeros(values.shape, "complex")
    for comp in np.arange(num_comps):
      ft_values[..., comp] = scipy.fft.fftn(values[..., comp])
    # end
    if psd:
      for i in range(0, num_dims):
        freq[i] = freq[i][:N[i]//2]
      if num_dims == 2:
        ft_values = np.abs(ft_values[:N[0]//2, :N[1]//2, :])**2
        # If only 2D, append third dummy index for ease of logic
        freq.append(0)
      elif num_dims == 3:
        ft_values = np.abs(ft_values[:N[0]//2, :N[1]//2, :N[2]//2, :])**2
      else:
        raise ValueError("Only 1D, 2D, and 3D data are currently supported.")
      # end
      if iso:
        nkpolar = int(np.sqrt(np.sum(N[:] ** 2)))
        nkx = N[0]//2
        nky = N[1]//2
        nkz = N[2]//2
        kx = freq[0]
        ky = freq[1]
        kz = freq[2]
        akp, nbin, polar_index, _ = init_polar(nkx, nky, nkz, kx, ky, kz, nkpolar)
        fft_iso = np.zeros((nkpolar, num_comps))
        for comp in np.arange(num_comps):
          fft_iso[:, comp] = polar_isotropic(nkpolar, nkx, nky, nkz, polar_index,
              nbin, ft_values[..., comp], kx, ky, kz)
        # end
        # Return isotropic spectra and 1D isotropic ks
        if overwrite:
          data.push([akp], fft_iso)
        return [akp], fft_iso
        # end
      # end
    # end

    if overwrite and not iso:
      data.push(freq, ft_values)
    return freq, ft_values
    # end
  # end
