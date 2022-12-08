#!/usr/bin/env python
"""
Postgkyl module for computing primitive variables from conservative variables
"""
import numpy as np

def get_density(in_data=None,
                in_grid=None, in_values=None, 
                overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid = in_grid
  out_values = in_values[..., 0, np.newaxis]
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_vx(in_data=None,
           in_grid=None, in_values=None, 
           overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  rhovx = in_values[..., 1, np.newaxis]
  out_values = rhovx/rho
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_vy(in_data=None,
           in_grid=None, in_values=None, 
           overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  rhovy = in_values[..., 2, np.newaxis]
  out_values = rhovy/rho
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_vz(in_data=None,
           in_grid=None, in_values=None, 
           overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  rhovz = in_values[..., 3, np.newaxis]
  out_values = rhovz/rho
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_vi(in_data=None,
           in_grid=None, in_values=None, 
           overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  rhovi = in_values[..., 1:4, np.newaxis]
  out_values = rhovi/rho
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pxx(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vx = get_vx(in_data, in_grid, in_values)

  out_values = in_values[..., 4, np.newaxis] - rho*vx*vx
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pxy(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vx = get_vx(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)

  out_values = in_values[..., 5, np.newaxis] - rho*vx*vy
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pxz(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vx = get_vx(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)

  out_values = in_values[..., 6, np.newaxis] - rho*vx*vz
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pyy(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)

  out_values = in_values[..., 7, np.newaxis] - rho*vy*vy
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pyz(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)

  out_values = in_values[..., 8, np.newaxis] - rho*vy*vz
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pzz(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)

  out_values = in_values[..., 9, np.newaxis] - rho*vz*vz
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_pij(in_data=None,
            in_grid=None, in_values=None, 
            overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_values = np.zeros(in_values[..., 4:10].shape)
  
  out_grid, pxx = get_pxx(in_data, in_grid, in_values)
  out_grid, pxy = get_pxy(in_data, in_grid, in_values)
  out_grid, pxz = get_pxz(in_data, in_grid, in_values)
  out_grid, pyy = get_pyy(in_data, in_grid, in_values)
  out_grid, pyz = get_pyz(in_data, in_grid, in_values)
  out_grid, pzz = get_pzz(in_data, in_grid, in_values)
                    
  out_values[..., 0] = np.squeeze(pxx)
  out_values[..., 1] = np.squeeze(pxy)
  out_values[..., 2] = np.squeeze(pxz)
  out_values[..., 3] = np.squeeze(pyy)
  out_values[..., 4] = np.squeeze(pyz)
  out_values[..., 5] = np.squeeze(pzz)  

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_p(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, numMom=None, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  if numMom is None:
    if in_data.getNumComps() == 5:
      numMom = 5
    elif in_data.getNumComps() == 10:
      numMom = 10
    else:
      raise ValueError("Number of components appears to be {:d};"
                       "it needs to be specified using 'numMom' "
                       "(5 or 10)".format(in_data.getNumComps()))
    #end
  #end

  if numMom == 5:
    out_grid, rho = get_density(in_data, in_grid, in_values)
    out_grid, vx = get_vx(in_data, in_grid, in_values)
    out_grid, vy = get_vy(in_data, in_grid, in_values)
    out_grid, vz = get_vz(in_data, in_grid, in_values)
     
    out_values = (gasGamma - 1)*(in_values[..., 4, np.newaxis] - 0.5*rho*(vx**2 + vy**2 + vz**2))
  elif numMom == 10:
    out_grid, pxx = get_pxx(in_data, in_grid, in_values)
    out_grid, pyy = get_pyy(in_data, in_grid, in_values)
    out_grid, pzz = get_pzz(in_data, in_grid, in_values)

    out_values = (pxx + pyy + pzz) / 3.0
  #end

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_ke(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, numMom=None, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  if numMom is None:
    if in_data.getNumComps() == 5:
      numMom = 5
    elif in_data.getNumComps() == 10:
      numMom = 10
    else:
      raise ValueError("Number of components appears to be {:d};"
                       "it needs to be specified using 'numMom' "
                       "(5 or 10)".format(in_data.getNumComps()))
    #end
  #end

  if numMom == 5:
    out_grid, pr = get_p(in_data, in_grid, in_values, gasGamma, numMom)

    out_values = in_values[..., 4, np.newaxis] - pr/(gasGamma - 1)
  elif numMom == 10:
    out_grid, rho = get_density(in_data, in_grid, in_values)
    out_grid, vx = get_vx(in_data, in_grid, in_values)
    out_grid, vy = get_vy(in_data, in_grid, in_values)
    out_grid, vz = get_vz(in_data, in_grid, in_values)

    out_values = 0.5*rho*(vx**2 + vy**2 + vz**2)
  #end

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end  

def get_sound(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, numMom=None, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, pr = get_p(in_data, in_grid, in_values, gasGamma, numMom)

  out_values = np.sqrt(gasGamma*pr/rho)

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end 

def get_mach(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, numMom=None, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, vx = get_vx(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)
  out_grid, cs = get_sound(in_data, in_grid, in_values, gasGamma, numMom)

  out_values = np.sqrt(vx**2+vy**2+vz**2)/cs

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_mhd_Bx(in_data=None,
               in_grid=None, in_values=None, 
               overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid = in_grid
  out_values = in_values[..., 5, np.newaxis]
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_mhd_By(in_data=None,
               in_grid=None, in_values=None, 
               overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid = in_grid
  out_values = in_values[..., 6, np.newaxis]
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_mhd_Bz(in_data=None,
               in_grid=None, in_values=None, 
               overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid = in_grid
  out_values = in_values[..., 7, np.newaxis]
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_mhd_Bi(in_data=None,
           in_grid=None, in_values=None, 
           overwrite=False):

  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end
  out_grid = in_grid
  out_values = in_values[..., 5:8, np.newaxis]
  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end

def get_mhd_mag_p(in_data=None,
          in_grid=None, in_values=None, 
          mu0=1.0, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, Bx = get_mhd_Bx(in_data, in_grid, in_values)
  out_grid, By = get_mhd_By(in_data, in_grid, in_values)
  out_grid, Bz = get_mhd_Bz(in_data, in_grid, in_values)

  out_values = 0.5*(Bx**2 + By**2 + Bz**2)/mu0

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end  

def get_mhd_p(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, mu0=1.0, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, vx = get_vx(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)
  out_grid, mag_p = get_mhd_mag_p(in_data, in_grid, in_values, mu0)

  out_values = (gasGamma - 1)*(in_values[..., 4, np.newaxis] - 0.5*rho*(vx**2 + vy**2 + vz**2) - mag_p)

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end   

def get_mhd_sound(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, mu0=1.0, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, rho = get_density(in_data, in_grid, in_values)
  out_grid, pr = get_mhd_p(in_data, in_grid, in_values, gasGamma, mu0)

  out_values = np.sqrt(gasGamma*pr/rho)

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end  

def get_mhd_mach(in_data=None,
          in_grid=None, in_values=None, 
          gasGamma=5.0/3.0, mu0=1.0, 
          overwrite=False):
  if in_data:
    in_grid = in_data.getGrid()
    in_values = in_data.getValues()
  #end

  out_grid, vx = get_vx(in_data, in_grid, in_values)
  out_grid, vy = get_vy(in_data, in_grid, in_values)
  out_grid, vz = get_vz(in_data, in_grid, in_values)
  out_grid, cs = get_mhd_sound(in_data, in_grid, in_values, gasGamma, mu0)

  out_values = np.sqrt(vx**2+vy**2+vz**2)/cs

  if overwrite:
    in_data.push(out_grid, out_values)
  #end
  return out_grid, out_values
#end