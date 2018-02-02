import numpy as np

def getDensity(values):
    return values[..., 0, np.newaxis]

def getVx(values):
    return values[..., 1, np.newaxis] / values[..., 0, np.newaxis]

def getVy(values):
    return values[..., 2, np.newaxis] / values[..., 0, np.newaxis]

def getVz(values):
    return values[..., 3, np.newaxis] / values[..., 0, np.newaxis]

def getV(values):
    return values[..., 1:4] / values[..., 0, np.newaxis]

def getPxx(values):
    r = getDensity(values)
    vx = getVx(values)
    return q[..., 4, np.newaxis] - r*vx*vx

def getPxy(values):
    r = getDensity(values)
    vx = getVx(values)
    vy = getVy(values)
    return values[..., 5, np.newaxis] - r*vx*vy

def getPxz(values):
    r = getDensity(values)
    vx = getVx(values)
    vz = getVz(values)
    return values[..., 6, np.newaxis] - r*vx*vz

def getPyy(values):
    r = getDensity(values)
    vy = getVy(values)
    return values[..., 7, np.newaxis] - r*vy*vy

def getPyz(values):
    r = getDensity(values)
    vy = getVy(values)
    vz = getVz(values)
    return values[..., 8, np.newaxis] - r*vy*vz

def getPzz(values):
    r = getDensity(values)
    vz = getVz(values)
    return values[..., 9, np.newaxis] - r*vz*vz

def getPij(values):
    tmp = np.copy(values[..., 4:10])
    tmp[..., 0] = getPxx(values)
    tmp[..., 1] = getPxy(values)
    tmp[..., 2] = getPxz(values)
    tmp[..., 3] = getPyy(values)
    tmp[..., 4] = getPyz(values)
    tmp[..., 5] = getPzz(values)
    return tmp

def getPii(values, gasGamma=5.0/3, numMom=None):
    if numMom is None:
        if values.shape[-1] == 5:
            numMom = 5
        elif values.shape[-1] == 10:
            numMom = 10
        else:
            raise ValueError("Number of components appears to be {:d};"
                             "it needs to be specified using 'numMom' "
                             "(5 or 10)".format(values.shape[-1]))

    if numMom == 5:
        p = (gasGamma - 1) \
            * (values[..., 4, np.newaxis] \
               - 0.5*(values[..., 1, np.newaxis]**2 
                      + values[..., 2, np.newaxis]**2
                      + values[..., 3, np.newaxis]**2) / getDensity(values))
    elif numMom == 10:
        p = (getPxx(values) + getPyy(values) + getPzz(values)) / 3.0
    return p
