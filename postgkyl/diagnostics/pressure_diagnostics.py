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
import postgkyl.diagnostics as diag

def getPPar(dataSpecies, dataField, overwrite=False):

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

def getPPerp(dataSpecies, dataField, overwrite=False):

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

def getAgyro(dataSpecies, dataField, measure="swisdak", overwrite=False):

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
 