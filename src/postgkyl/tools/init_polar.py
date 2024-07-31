import numpy as np


def init_polar(nkx, nky, nkz, kx, ky, kz, nkpolar):
  # if 2D, nkz and kz = 0

  if nkpolar == 0:
    akp = []
    nbin = 0
    polar_index = []
    akplim = []
  elif nkz == 0:
    nbin = np.zeros(nkpolar)  # Number of kx,ky in each polar bins
    polar_index = np.zeros((nkx, nky), dtype=int)  # Polar index to simplify binning
    if nkx == 1 & nky == 1:
      dkp = 0
    elif nkx == 1:
      dkp = ky[1]
    elif nky == 1:
      dkp = kx[1]
    else:
      dkp = max(kx[1], ky[1])
    akp = (np.linspace(1, nkpolar, nkpolar)) * dkp  # Kperp grid
    akplim = dkp / 2 + (np.linspace(0, nkpolar, nkpolar + 1))*dkp  # Bin limits
    # Re-written to avoid loops. Necessary for large grids.
    [kxg, kyg] = np.meshgrid(
        ky, kx
    )  # Deal with meshgrid weirdness (so do not have to transpose)
    kp = np.sqrt(kxg**2 + kyg**2)
    pn = np.where(kp >= akplim[nkpolar])
    polar_index[pn[0], pn[1]] = nkpolar - 1
    nbin[nkpolar - 1] = nbin[nkpolar - 1] + len(pn[0])
    for ik in range(0, nkpolar):
      pn = np.where((kp < akplim[ik + 1]) & (kp >= akplim[ik]))
      polar_index[pn[0], pn[1]] = ik
      nbin[ik] = nbin[ik] + len(pn[0])
  else:
    # 3D data
    nbin = np.zeros(nkpolar)
    polar_index = np.zeros((nkx, nky, nkz), dtype=int)
    if nkx == 1 & nky == 1 & nkz == 1:
      dkp = 0
    elif nkx == 1:
      dkp = max(ky[1], kz[1])
    elif nky == 1:
      dkp = max(kx[1], kz[1])
    elif nkz == 1:
      dkp = max(kx[1], ky[1])
    else:
      dkp = max(kx[1], ky[1], kz[1])
    akp = (np.linspace(1, nkpolar, nkpolar)) * dkp  # kperp grid
    akplim = dkp / 2 + (np.linspace(0, nkpolar, nkpolar + 1)) * dkp  # bin limits
    # Re-written to avoid loops
    [kxg, kyg, kzg] = np.meshgrid(ky, kx, kz)
    kp = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    pn = np.where(kp >= akplim[nkpolar])
    polar_index[pn[0], pn[1], pn[2]] = nkpolar - 1
    nbin[nkpolar - 1] = nbin[nkpolar - 1] + len(pn[0])
    for ik in range(0, nkpolar):
      pn = np.where((kp < akplim[ik + 1]) & (kp >= akplim[ik]))
      polar_index[pn[0], pn[1], pn[2]] = ik
      nbin[ik] = nbin[ik] + len(pn[0])

  return akp, nbin, polar_index, akplim
