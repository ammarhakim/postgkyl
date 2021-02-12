#!/usr/bin/env python
"""
Postgkyl module for computing primitive variables from conservative variables
"""
import numpy as np

def getDensity(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 0].shape)
    out = out[..., np.newaxis]

    out[..., 0] = values[..., 0]

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVx(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 1].shape)
    out = out[..., np.newaxis]

    out[..., 0] =  values[..., 1] / values[..., 0]

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVy(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 2].shape)
    out = out[..., np.newaxis]

    out[..., 0] = values[..., 2] / values[..., 0]

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVz(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 3].shape)
    out = out[..., np.newaxis]

    out[..., 0] = values[..., 3] / values[..., 0]

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVi(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 1:4].shape)

    out[..., 0:3] = values[..., 1:4] / values[..., 0]

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxx(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 4].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    out[..., 0] = values[..., 4] - rho*vx*vx

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxy(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 5].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    grid, vy = getVy(data)

    out[..., 0] = values[..., 5] - rho*vx*vy

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxz(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 6].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    grid, vz = getVz(data)

    out[..., 0] = values[..., 6] - rho*vx*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPyy(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 7].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vy = getVy(data)

    out[..., 0] = values[..., 7] - rho*vy*vy

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPyz(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 8].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vy = getVy(data)
    grid, vz = getVz(data)

    out[..., 0] = values[..., 8] - rho*vy*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPzz(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 9].shape)
    out = out[..., np.newaxis]

    grid, rho = getDensity(data)
    grid, vz = getVz(data)

    out[..., 0] = values[..., 9] - rho*vz*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPij(data, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 4:10].shape)

    grid, Pxx = getPxx(data)
    grid, Pxy = getPxy(data)
    grid, Pxz = getPxz(data)
    grid, Pyy = getPyy(data)
    grid, Pyz = getPyz(data)
    grid, Pzz = getPzz(data)
                        
    out[..., 0] = np.squeeze(Pxx)
    out[..., 1] = np.squeeze(Pxy)
    out[..., 2] = np.squeeze(Pxz)
    out[..., 3] = np.squeeze(Pyy)
    out[..., 4] = np.squeeze(Pyz)
    out[..., 5] = np.squeeze(Pzz)

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getP(data, gasGamma=5.0/3.0, numMom=None, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 0].shape)
    out = out[..., np.newaxis]

    if numMom is None:
        if data.getNumComps() == 5:
            numMom = 5
        elif data.getNumComps() == 10:
            numMom = 10
        else:
            raise ValueError("Number of components appears to be {:d};"
                             "it needs to be specified using 'numMom' "
                             "(5 or 10)".format(data.getNumComps()))
        #end
    #end

    if numMom == 5:
        grid, rho = getDensity(data)
        grid, vx = getVx(data)  
        grid, vy = getVy(data)  
        grid, vz = getVz(data)       
        out[..., 0] = (gasGamma - 1)*(values[..., 4] - 0.5*rho[..., 0]*(vx[..., 0]**2 + vy[..., 0]**2 + vz[..., 0]**2))
    elif numMom == 10:
        grid, Pxx = getPxx(data)  
        grid, Pyy = getPyy(data)  
        grid, Pzz = getPzz(data) 
        out[..., 0] = (Pxx[..., 0] + Pyy[..., 0] + Pzz[..., 0]) / 3.0
    #end

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getKE(data, gasGamma=5.0/3, numMom=None, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 0].shape)
    out = out[..., np.newaxis]

    grid, pr = getP(data, gasGamma, numMom)

    out[..., 0] = values[..., 4] - pr[..., 0]/(gasGamma-1)

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getMach(data, gasGamma=5.0/3, numMom=None, overwrite=False):
    grid = data.getGrid()
    values = data.getValues()
    out = np.zeros(values[..., 0].shape)
    out = out[..., np.newaxis]


    grid, rho = getDensity(data)
    grid, vx = getVx(data)  
    grid, vy = getVy(data)  
    grid, vz = getVz(data) 
    grid, pr = getP(data, gasGamma, numMom)

    # Sound speed cs = sqrt(gasGamma*pr/rho)
    out[..., 0] = np.sqrt(vx[..., 0]**2+vy[..., 0]**2+vz[..., 0]**2)/np.sqrt(gasGamma*pr[..., 0]/rho[..., 0])

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end
