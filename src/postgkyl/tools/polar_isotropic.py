import numpy as np


def polar_isotropic(nkpolar, nkx, nky, nkz, polar_index, nbin, fft_matrix, kx, ky, kz):
  # if 2D, then nkz = kz = 0

  fft_isok = np.zeros(nkpolar)
  if nkz == 0:
    for i in range(0, nkx):
      for j in range(0, nky):
        fft_isok[polar_index[i, j]] = fft_isok[polar_index[i, j]] + fft_matrix[i, j]
  else:
    for i in range(0, nkx):
      for j in range(0, nky):
        for k in range(0, nkz):
          fft_isok[polar_index[i, j, k]] = fft_isok[polar_index[i, j, k]] + fft_matrix[i, j, k]

  fft_isok = fft_isok / nbin[:]
  return fft_isok
