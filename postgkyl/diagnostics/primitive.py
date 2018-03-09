import numpy as np

def getDensity(data, stack=False):
    values = data.peakValues()
    out = values[..., 0, np.newaxis]
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getVx(data, stack=False):
    values = data.peakValues()
    out =  values[..., 1, np.newaxis] / values[..., 0, np.newaxis]
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getVy(data, stack=False):
    values = data.peakValues()
    out = values[..., 2, np.newaxis] / values[..., 0, np.newaxis]
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getVz(data, stack=False):
    values = data.peakValues()
    out = values[..., 3, np.newaxis] / values[..., 0, np.newaxis]
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getVi(data, stack=False):
    values = data.peakValues()
    out = values[..., 1:4] / values[..., 0, np.newaxis]
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPxx(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vx = getVx(data)
    out = values[..., 4, np.newaxis] - r*vx*vx
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPxy(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vx = getVx(data)
    vy = getVy(data)
    out = values[..., 5, np.newaxis] - r*vx*vy
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPxz(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vx = getVx(data)
    vz = getVz(data)
    out = values[..., 6, np.newaxis] - r*vx*vz
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPyy(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vy = getVy(data)
    return values[..., 7, np.newaxis] - r*vy*vy
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPyz(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vy = getVy(data)
    vz = getVz(data)
    out = values[..., 8, np.newaxis] - r*vy*vz
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPzz(data, stack=False):
    values = data.peakValues()
    r = getDensity(data)
    vz = getVz(data)
    out = values[..., 9, np.newaxis] - r*vz*vz
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getPij(data, stack=False):
    values = data.peakValues()
    out = np.copy(values[..., 4:10])
    out[..., 0] = getPxx(data)
    out[..., 1] = getPxy(data)
    out[..., 2] = getPxz(data)
    out[..., 3] = getPyy(data)
    out[..., 4] = getPyz(data)
    out[..., 5] = getPzz(data)
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out

def getP(data, gasGamma=5.0/3, numMom=None, stack=False):
    values = data.peakValues()
    if numMom is None:
        if data.getNumComps() == 5:
            numMom = 5
        elif data.getNumComps() == 10:
            numMom = 10
        else:
            raise ValueError("Number of components appears to be {:d};"
                             "it needs to be specified using 'numMom' "
                             "(5 or 10)".format(data.getNumComps()))

    if numMom == 5:
        out = (gasGamma - 1) \
              * (values[..., 4, np.newaxis] \
                 - 0.5*(getVx(data)**2 
                        + getVy(data)**2
                        + getVz(data)**2) * getDensity(data))
    elif numMom == 10:
        out = (getPxx(data) + getPyy(data) + getPzz(data)) / 3.0
    if stack:
        data.pushGrid()
        data.pushValues(out)
    else:
        return out
