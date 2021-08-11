#!/usr/bin/env python
"""
Postgkyl module for computing primitive variables from conservative variables
"""
import numpy as np

def getDensity(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()
    out = values[..., 0, np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVx(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()
    out =  values[..., 1, np.newaxis] / values[..., 0, np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVy(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()
    out =  values[..., 2, np.newaxis] / values[..., 0, np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVz(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()
    out =  values[..., 3, np.newaxis] / values[..., 0, np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getVi(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()
    out = values[..., 1:4] / values[..., 0, np.newaxis]
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxx(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    out = values[..., 4, np.newaxis] - rho*vx*vx

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxy(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    grid, vy = getVy(data)

    out = values[..., 5, np.newaxis] - rho*vx*vy

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPxz(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vx = getVx(data)
    grid, vz = getVz(data)

    out = values[..., 6, np.newaxis] - rho*vx*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPyy(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vy = getVy(data)

    out = values[..., 7, np.newaxis] - rho*vy*vy

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPyz(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vy = getVy(data)
    grid, vz = getVz(data)

    out = values[..., 8, np.newaxis] - rho*vy*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPzz(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vz = getVz(data)

    out = values[..., 9, np.newaxis] - rho*vz*vz

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPij(data, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
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

def getP(data, gasGamma=5.0/3.0, numMom=None, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

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
        out = (gasGamma - 1)*(values[..., 4, np.newaxis] - 0.5*rho*(vx**2 + vy**2 + vz**2))
    elif numMom == 10:
        grid, Pxx = getPxx(data)  
        grid, Pyy = getPyy(data)  
        grid, Pzz = getPzz(data) 
        out = (Pxx + Pyy + Pzz) / 3.0
    #end

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getKE(data, gasGamma=5.0/3, numMom=None, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    grid = data.getGrid()
    values = data.getValues()

    grid, pr = getP(data, gasGamma, numMom)

    out = values[..., 4, np.newaxis] - pr/(gasGamma-1)

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getSound(data, gasGamma=5.0/3, numMom=None, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid = data.getGrid()
    values = data.getValues()

    grid, rho = getDensity(data)
    grid, vx = getVx(data)  
    grid, vy = getVy(data)  
    grid, vz = getVz(data) 
    grid, pr = getP(data, gasGamma, numMom)

    # Sound speed cs = sqrt(gasGamma*pr/rho)
    out = np.sqrt(gasGamma*pr/rho)

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getMach(data, gasGamma=5.0/3, numMom=None, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid = data.getGrid()
    values = data.getValues()

    grid, vx = getVx(data)  
    grid, vy = getVy(data)  
    grid, vz = getVz(data) 
    grid, cs = getSound(data, gasGamma, numMom)

    # Sound speed cs = sqrt(gasGamma*pr/rho)
    out = np.sqrt(vx**2+vy**2+vz**2)/cs

    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end
