import numpy as np

def add(inGrid, inValues):
    if inGrid[0] != []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[0] + inValues[1]
    return [outGrid], [outValues]


def subtract(inGrid, inValues):
    if inGrid[0] != []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] - inValues[0]
    return [outGrid], [outValues]


def mult(inGrid, inValues):
    if inGrid[0] != []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] * inValues[0]
    return [outGrid], [outValues]


def divide(inGrid, inValues):
    if inGrid[0] != []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] / inValues[0]
    return [outGrid], [outValues]


def sqrt(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.sqrt(inValues[0])
    return [outGrid], [outValues]


def absolute(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.abs(inValues[0])
    return [outGrid], [outValues]


def log(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.log(inValues[0])
    return [outGrid], [outValues]


def log10(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.log10(inValues[0])
    return [outGrid], [outValues] 


def minimum(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.min(inValues[0]))
    return [outGrid], [outValues]


def maximum(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.max(inValues[0]))
    return [outGrid], [outValues] 


def mean(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.mean(inValues[0]))
    return [outGrid], [outValues]


def power(inGrid, inValues):
    if inGrid[0] != []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = np.pow(inValues[1], inValues[0])
    return [outGrid], [outValues] 


def grad(inGrid, inValues):
    outGrid = inGrid[1]
    ax = inValues[0]
    if type(ax) == str and ':' in ax:
        tmp = ax.split(':')
        if tmp[0] == '':
            lo = None
        else:
            lo = int(tmp[0])
        if tmp[1] == '':
            up = None
        else:
            up = int(tmp[1])
        ax = slice(lo, up)
    else:
        ax = int(ax)
        lo = ax
        up = ax+1

    numDims = up-lo
    outShape = list(inValues[1].shape)
    numComps = inValues[1].shape[-1]
    outShape[-1] = outShape[-1]*numDims
    outValues = np.zeros(outShape)
    
    for cnt, d in enumerate(range(lo, up)):
        zc = 0.5*(inGrid[1][d][1:] + inGrid[1][d][:-1]) # get cell centered values
        outValues[...,cnt*numComps:(cnt+1)*numComps] = np.gradient(inValues[1], zc, edge_order=2, axis=d)
    return [outGrid], [outValues]


def divergence(inGrid, inValues):
    outGrid = inGrid[0]
    numDims = len(inGrid[0])
    numComps = inValues[0].shape[-1]
    if numComps != numDims:
        raise ValueError("Number of components does not correspond to the number of dimensions")

    outShape = list(inValues[0].shape)
    outShape[-1] = 1
    outValues = np.zeros(outShape)
    for d in range(numDims):
        zc = 0.5*(inGrid[0][d][1:] + inGrid[0][d][:-1]) # get cell centered values
        outValues[..., 0] = outValues[..., 0] + np.gradient(inValues[0][..., d], zc, edge_order=2, axis=d)
    return [outGrid], [outValues]


def curl(inGrid, inValues):
    outGrid = inGrid[0]
    numDims = len(inGrid[0])
    numComps = inValues[0].shape[-1]
    if numDims != 2 and numDims !=3:
        raise ValueError("Number of dimensions needs to be either 2 or 3")
    if numComps != numDims:
        raise ValueError("Number of components does not correspond to the number of dimensions")

    outShape = list(inValues[0].shape)
    if numDims == 2:
        outShape[-1] = 1
    outValues = np.zeros(outShape)
    if numDims == 2:
        zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
        zc1 = 0.5*(inGrid[0][1][1:] + inGrid[0][1][:-1])
        outValues[..., 0] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)
    else:
        zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
        zc1 = 0.5*(inGrid[0][1][1:] + inGrid[0][1][:-1])
        zc2 = 0.5*(inGrid[0][2][1:] + inGrid[0][2][:-1])
        outValues[..., 0] = np.gradient(inValues[0][..., 2], zc1, edge_order=2, axis=1) - np.gradient(inValues[0][..., 1], zc2, edge_order=2, axis=2)
        outValues[..., 1] = np.gradient(inValues[0][..., 0], zc2, edge_order=2, axis=2) - np.gradient(inValues[0][..., 2], zc0, edge_order=2, axis=0)
        outValues[..., 2] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)

    return [outGrid], [outValues]


cmds = { '+' : { 'numIn' : 2, 'numOut' : 1, 'func' : add }, 
         '-' : { 'numIn' : 2, 'numOut' : 1, 'func' : subtract },
         '*' : { 'numIn' : 2, 'numOut' : 1, 'func' : mult },
         '/' : { 'numIn' : 2, 'numOut' : 1, 'func' : divide },
         'sqrt' : { 'numIn' : 1, 'numOut' : 1, 'func' : sqrt },
         'abs' : { 'numIn' : 1, 'numOut' : 1, 'func' : absolute },
         'log' : { 'numIn' : 1, 'numOut' : 1, 'func' : log },
         'log10' : { 'numIn' : 1, 'numOut' : 1, 'func' : log10 },
         'max' : { 'numIn' : 1, 'numOut' : 1, 'func' : maximum },
         'min' : { 'numIn' : 1, 'numOut' : 1, 'func' : minimum },
         'mean' : { 'numIn' : 1, 'numOut' : 1, 'func' : mean },
         'pow' : { 'numIn' : 2, 'numOut' : 1, 'func' : power },
         'grad' : { 'numIn' : 2, 'numOut' : 1, 'func' : grad },
         'div' : { 'numIn' : 1, 'numOut' : 1, 'func' : divergence },
         'curl' : { 'numIn' : 1, 'numOut' : 1, 'func' : curl },
}
