#!/usr/bin/env python
"""
Postgkyl module for pressure tensor diagnostics
Diagnostics include:
    Pressure parallel to the magnetic field
    Pressure perpendicular to the magnetic field
    Agyrotropy (either Frobenius or Swisdak measure)
    Firehose instability threshold
"""
import numpy as np
from .. import diagnostics as diag

def getDivP(dataSpecies, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end
    
    if dataSpecies.getNumComps() == 10:
        numMom = 10
    else:
        raise ValueError("Number of components appears to be {:d};"
                         "it needs to be 10".format(dataSpecies.getNumComps()))
    #end
    grid, Pxx = diag.primitive.getPxx(dataSpecies)
    grid, Pxy = diag.primitive.getPxy(dataSpecies)
    grid, Pxz = diag.primitive.getPxz(dataSpecies)
    grid, Pyy = diag.primitive.getPyy(dataSpecies)
    grid, Pyz = diag.primitive.getPyz(dataSpecies)
    grid, Pzz = diag.primitive.getPzz(dataSpecies)

    outShape = list(Pxx.shape)
    outShape[-1] = outShape[-1]*3
    outValues = np.zeros(outShape)
    if dataSpecies.getNumDims() == 1:
        dx = grid[0][1] - grid[0][0]
        outValues[..., 0] = np.gradient(Pxx[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 1] = np.gradient(Pxy[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 2] = np.gradient(Pxz[..., 0], dx, edge_order=2, axis=0)
    elif dataSpecies.getNumDims() == 2:
        dx = grid[0][1] - grid[0][0]
        dy = grid[1][1] - grid[1][0]
        outValues[..., 0] = np.gradient(Pxx[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pxy[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 1] = np.gradient(Pxy[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pyy[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 2] = np.gradient(Pxz[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pyz[..., 0], dy, edge_order=2, axis=1)
    else:
        dx = grid[0][1] - grid[0][0]
        dy = grid[1][1] - grid[1][0]
        dz = grid[2][1] - grid[2][0]
        outValues[..., 0] = np.gradient(Pxx[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pxy[..., 0], dy, edge_order=2, axis=1) + np.gradient(Pxz[..., 0], dz, edge_order=2, axis=2)
        outValues[..., 1] = np.gradient(Pxy[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pyy[..., 0], dy, edge_order=2, axis=1) + np.gradient(Pyz[..., 0], dz, edge_order=2, axis=2)
        outValues[..., 2] = np.gradient(Pxz[..., 0], dx, edge_order=2, axis=0) + np.gradient(Pyz[..., 0], dy, edge_order=2, axis=1) + np.gradient(Pzz[..., 0], dz, edge_order=2, axis=2)
    #end
    if overwrite:
        data.push(grid, outValues)
    else:
        return grid, outValues
    #end
#end    

def getGradU(dataSpecies, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid, ux = diag.primitive.getVx(dataSpecies)
    grid, uy = diag.primitive.getVy(dataSpecies)
    grid, uz = diag.primitive.getVz(dataSpecies) 

    outShape = list(ux.shape)
    outShape[-1] = outShape[-1]*3*dataSpecies.getNumDims()
    outValues = np.zeros(outShape)
    if dataSpecies.getNumDims() == 1:
        dx = grid[0][1] - grid[0][0]
        outValues[..., 0] = np.gradient(ux[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 1] = np.gradient(uy[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 2] = np.gradient(uz[..., 0], dx, edge_order=2, axis=0)
    elif dataSpecies.getNumDims() == 2:
        dx = grid[0][1] - grid[0][0]
        dy = grid[1][1] - grid[1][0]
        outValues[..., 0] = np.gradient(ux[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 1] = np.gradient(uy[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 2] = np.gradient(uz[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 3] = np.gradient(ux[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 4] = np.gradient(uy[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 5] = np.gradient(uz[..., 0], dy, edge_order=2, axis=1)        
    else:
        dx = grid[0][1] - grid[0][0]
        dy = grid[1][1] - grid[1][0]
        dz = grid[2][1] - grid[2][0]
        outValues[..., 0] = np.gradient(ux[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 1] = np.gradient(uy[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 2] = np.gradient(uz[..., 0], dx, edge_order=2, axis=0)
        outValues[..., 3] = np.gradient(ux[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 4] = np.gradient(uy[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 5] = np.gradient(uz[..., 0], dy, edge_order=2, axis=1)
        outValues[..., 6] = np.gradient(ux[..., 0], dz, edge_order=2, axis=2)
        outValues[..., 7] = np.gradient(uy[..., 0], dz, edge_order=2, axis=2)
        outValues[..., 8] = np.gradient(uz[..., 0], dz, edge_order=2, axis=2)    
    #end
    if overwrite:
        data.push(grid, outValues)
    else:
        return grid, outValues
    #end
#end

def pDelU(dataSpecies, gasGamma=5.0/3.0, overwrite=False, stack=False): 
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid, gradU = getGradU(dataSpecies)
    if dataSpecies.getNumComps() == 5:
        grid, ux = diag.primitive.getVx(dataSpecies)
        grid, uy = diag.primitive.getVy(dataSpecies)
        grid, p = diag.primitive.getP(dataSpecies, gasGamma)
        dx = grid[0][1] - grid[0][0]
        dy = grid[1][1] - grid[1][0]
        out = p*(np.gradient(ux, dx, edge_order=2, axis=0) + np.gradient(uy, dy, edge_order=2, axis=1)) #+ gradU[..., 8])
        # for i in range(0, dataSpecies.getNumDims()):
        #     out[...,0] += gradU[...,i+i*3]*p[...,0]
    elif dataSpecies.getNumComps() == 10:
        grid, Pij = diag.primitive.getPij(dataSpecies)
        ctr = 0
        for i in range(0, dataSpecies.getNumDims()):
            out[...,0] += Pij[...,0+i]*gradU[...,0+i*3] + Pij[...,1+i+ctr]*gradU[...,1+i*3] + Pij[...,2+i+ctr]*gradU[...,2+i*3]
            ctr = 1 # Additional increment needed to fetch correct Pij indicies (P_yy, P_yz, P_zz == components 3, 4, 5 respectively)
    else:
        raise ValueError("Number of components appears to be {:d};"
                         "it needs to be (5 or 10)".format(data.getNumComps()))
    #end
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end  

def pressureWork(dataSpecies, gasGamma=5.0/3.0, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid, pr = diag.primitive.getP(dataSpecies, gasGamma)
    grid, ke = diag.primitive.getKE(dataSpecies, gasGamma)
    grid, pdW = pDelU(dataSpecies, gasGamma)
    grid, ux = diag.primitive.getVx(dataSpecies)
    grid, uy = diag.primitive.getVy(dataSpecies)
    dx = grid[0][1] - grid[0][0]
    dy = grid[1][1] - grid[1][0]
    divU = np.gradient(ux, dx, edge_order=2, axis=0) + np.gradient(uy, dy, edge_order=2, axis=1)

    out = np.zeros(dataSpecies.getValues()[...,0:5].shape)
    out[..., 0] = np.squeeze(pr)/(gasGamma-1)
    out[..., 1] = np.squeeze(ke)
    out[..., 2] = np.squeeze(pdW)
    out[..., 3] = np.squeeze(pr)
    out[..., 4] = np.squeeze(divU)
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end  

def getPPar(dataSpecies, dataField, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    Bx = dataField.getValues()[..., 3]
    By = dataField.getValues()[..., 4]
    Bz = dataField.getValues()[..., 5]
    Bx = Bx[..., np.newaxis]
    By = By[..., np.newaxis]
    Bz = Bz[..., np.newaxis]

    out = np.zeros(Bx.shape)

    grid, Pxx = diag.primitive.getPxx(dataSpecies)
    grid, Pxy = diag.primitive.getPxy(dataSpecies)
    grid, Pxz = diag.primitive.getPxz(dataSpecies)
    grid, Pyy = diag.primitive.getPyy(dataSpecies)
    grid, Pyz = diag.primitive.getPyz(dataSpecies)
    grid, Pzz = diag.primitive.getPzz(dataSpecies)
    grid, magBsq = diag.magsq(dataField, coords='3:6')

    out = (Bx*Bx*Pxx + By*By*Pyy + Bz*Bz*Pzz + 2.0*(Bx*By*Pxy + Bx*Bz*Pxz + By*Bz*Pyz))/magBsq
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getPPerp(dataSpecies, dataField, overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    grid, PPar = getPPar(dataSpecies, dataField)
    
    out = np.zeros(PPar.shape)

    grid, Pxx = diag.primitive.getPxx(dataSpecies)
    grid, Pyy = diag.primitive.getPyy(dataSpecies)
    grid, Pzz = diag.primitive.getPzz(dataSpecies)

    out = (Pxx + Pyy + Pzz - PPar) / 2.0
    if overwrite:
        data.push(grid, out)
    else:
        return grid, out
    #end
#end

def getAgyro(dataSpecies, dataField, measure="swisdak", overwrite=False, stack=False):
    if stack:
        overwrite = stack
        print("Deprecation warning: The 'stack' parameter is going to be replaced with 'overwrite'")
    #end

    Bx = dataField.getValues()[..., 3]
    By = dataField.getValues()[..., 4]
    Bz = dataField.getValues()[..., 5]
    Bx = Bx[..., np.newaxis]
    By = By[..., np.newaxis]
    Bz = Bz[..., np.newaxis]

    out = np.zeros(Bx.shape)

    grid, Pxx = diag.primitive.getPxx(dataSpecies)
    grid, Pxy = diag.primitive.getPxy(dataSpecies)
    grid, Pxz = diag.primitive.getPxz(dataSpecies)
    grid, Pyy = diag.primitive.getPyy(dataSpecies)
    grid, Pyz = diag.primitive.getPyz(dataSpecies)
    grid, Pzz = diag.primitive.getPzz(dataSpecies)
    grid, magBsq = diag.magsq(dataField, coords='3:6')

    grid, PPar = getPPar(dataSpecies, dataField)
    grid, PPerp = getPPerp(dataSpecies, dataField)

    if measure.lower() == "swisdak":
        I1 = Pxx + Pyy + Pzz
        I2 = Pxx*Pyy + Pxx*Pzz + Pyy*Pzz - (Pxy*Pxy + Pxz*Pxz + Pyz*Pyz)

        # Note that this definition of Q uses the tensor algebra in Appendix A of Swisdak 2015.
        out = np.sqrt(1 - 4*I2/((I1 - PPar)*(I1 + 3*PPar))) 
        if overwrite:
            data.push(grid, out)
        else:
            return grid, out
        #end
    elif measure.lower() == "frobenius":
        Pixx = Pxx - (PPar*Bx*Bx/magBsq + PPerp*(1 - Bx*Bx/magBsq))
        Pixy = Pxy - (PPar*Bx*By/magBsq + PPerp*(0 - Bx*By/magBsq))
        Pixz = Pxz - (PPar*Bx*Bz/magBsq + PPerp*(0 - Bx*Bz/magBsq))
        Piyy = Pyy - (PPar*By*By/magBsq + PPerp*(1 - By*By/magBsq))
        Piyz = Pyz - (PPar*By*Bz/magBsq + PPerp*(0 - By*Bz/magBsq))
        Pizz = Pzz - (PPar*Bz*Bz/magBsq + PPerp*(1 - Bz*Bz/magBsq))
        out = np.sqrt(Pixx**2 + 2*Pixy**2 + 2*Pixz**2 + Piyy**2 + 2*Piyz**2 + Pizz**2) \
                / np.sqrt(2*PPerp**2 + 4*PPar*PPerp)
        if overwrite:
            data.push(grid, out)
        else:
            return grid, out
        #end
    else:
        raise ValueError("Measure specified is {:s};"
                 "it needs to be either 'swisdak' or 'frobenius'".format(measure.lower()))
    #end
#end
 
