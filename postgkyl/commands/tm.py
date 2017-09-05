import numpy as np

def getRho(q):
    return q[...,0]
def getU(q):
    return q[...,1]/q[...,0]
def getV(q):
    return q[...,2]/q[...,0]
def getW(q):
    return q[...,3]/q[...,0]

def getPxx(q):
    r = getRho(q)
    u = getU(q)
    return q[...,4] - r*u*u

def getPxy(q):
    r = getRho(q)
    u = getU(q)
    v = getV(q)
    return q[...,5] - r*u*v

def getPxz(q):
    r = getRho(q)
    u = getU(q)
    w = getW(q)
    return q[...,6] - r*u*w

def getPyy(q):
    r = getRho(q)
    v = getV(q)
    return q[...,7] - r*v*v

def getPyz(q):
    r = getRho(q)
    v = getV(q)
    w = getW(q)
    return q[...,8] - r*v*w

def getPzz(q):
    r = getRho(q)
    w = getW(q)
    return q[...,9] - r*w*w

def getVel(q):
    tmp = np.copy(q[..., 1:4])
    tmp[..., 0] = getU(q)
    tmp[..., 1] = getV(q)
    tmp[..., 2] = getW(q)
    return tmp

def getPressureTensor(q):
    tmp = np.copy(q[..., 4:10])
    tmp[..., 0] = getPxx(q)
    tmp[..., 1] = getPxy(q)
    tmp[..., 2] = getPxz(q)
    tmp[..., 3] = getPyy(q)
    tmp[..., 4] = getPyz(q)
    tmp[..., 5] = getPzz(q)
    return tmp

def getPressure(q):
    return (getPxx(q)+getPyy(q)+getPzz(q))/3.0
