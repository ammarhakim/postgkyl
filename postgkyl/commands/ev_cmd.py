import click
import numpy as np


def _gridCheck(grid0, grid1):
    if grid0 != [] and grid1 != []:
        cnt0, cnt1 = 0, 0
        for d in range(len(grid0)):
            if len(grid0[d]) == 1:
                cnt0 += 1
        for d in range(len(grid1)):
            if len(grid1[d]) == 1:
                cnt1 += 1

        if cnt0 < cnt1:
            return grid0
        else:
            return grid1
    elif grid0 != []:
        return grid0
    elif grid1 != []:
        return grid1
    else:
        return []


def add(inGrid, inValues):
    outGrid = _gridCheck(inGrid[0], inGrid[1])
    outValues = inValues[0] + inValues[1]
    return [outGrid], [outValues]


def subtract(inGrid, inValues):
    outGrid = _gridCheck(inGrid[0], inGrid[1])
    outValues = inValues[1] - inValues[0]
    return [outGrid], [outValues]


def mult(inGrid, inValues):
    outGrid = _gridCheck(inGrid[0], inGrid[1])
    outValues = inValues[1] * inValues[0]
    return [outGrid], [outValues]


def divide(inGrid, inValues):
    outGrid = _gridCheck(inGrid[0], inGrid[1])
    outValues = inValues[1] / inValues[0]
    return [outGrid], [outValues]


def sqrt(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.sqrt(inValues[0])
    return [outGrid], [outValues]

def psin(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.sin(inValues[0])
    return [outGrid], [outValues]

def pcos(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.cos(inValues[0])
    return [outGrid], [outValues]

def ptan(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.tan(inValues[0])
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
    outGrid = inGrid[1]
    outValues = np.power(inValues[1], inValues[0])
    return [outGrid], [outValues] 


def sq(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = inValues[0]**2
    return [outGrid], [outValues]

def exp(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.exp(inValues[0])
    return [outGrid], [outValues]


def length(inGrid, inValues):
    ax = int(inValues[0])
    length = inGrid[1][ax][-1] - inGrid[1][ax][0]
    if len(inGrid[1][ax]) == inValues[1].shape[ax]:
        length = length + inGrid[1][ax][1] - inGrid[1][ax][0]
    #end
    
    return [[]], [length]


def grad(inGrid, inValues):
    outGrid = inGrid[1]
    ax = inValues[0]
    if isinstance(ax, str) and ':' in ax:
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


def integrate(inGrid, inValues, avg=False):
    grid = inGrid[1].copy()
    values = np.array(inValues[1])

    axis = inValues[0]
    if isinstance(axis, float):
        axis = tuple([int(axis)])
    elif isinstance(axis, tuple):
        pass
    elif isinstance(axis, str):
        if len(axis.split(',')) > 1:
            axes = axis.split(',')
            axis = tuple([int(a) for a in axes])
        elif len(axis.split(':')) == 2:
            bounds = axis.split(':')
            axis = tuple(range(bouns[0], bounds[1]))
        elif axis == 'all':
            numDims = len(grid)
            axis = tuple(range(numDims))
    else:
            raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")

    dz = []
    for d, coord in enumerate(grid):
        dz.append(coord[1:] - coord[:-1])
        if len(coord) == values.shape[d]:
            dz[-1] = np.append(dz[-1], dz[-1][-1])

    # Integration assuming values are cell centered averages
    # Should work for nonuniform meshes
    for ax in sorted(axis, reverse=True):
        values = np.moveaxis(values, ax, -1)
        values = np.dot(values, dz[ax])
    #end
    for ax in sorted(axis):
        grid[ax] = np.array([0])
        values = np.expand_dims(values, ax)
        if avg:
            length = inGrid[1][ax][-1] - inGrid[1][ax][0]
            if len(inGrid[1][ax]) == inValues[1].shape[ax]:
                length = length + inGrid[1][ax][1] - inGrid[1][ax][0]
            #end
            values = values / length
        #end
    #end

    return [grid], [values]

def average(inGrid, inValues):
    return integrate(inGrid, inValues, True)

def divergence(inGrid, inValues):
    outGrid = inGrid[0]
    numDims = len(inGrid[0])
    numComps = inValues[0].shape[-1]
    if numComps > numDims:
        click.echo(click.style("WARNING in 'ev div': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). The last {:d} components of the vector will be disregarded.".format(numComps, numDims, numComps-numDims), fg='yellow'))

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

    outShape = list(inValues[0].shape)
    if numDims == 2:
        outShape[-1] = 1
    outValues = np.zeros(outShape)
    if numDims == 2:
        if numComps > 2:
            click.echo(click.style("WARNING in 'ev curl': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). Only the third component of curl will be calculated.".format(numComps, numDims), fg='yellow'))
        elif numComps < 2:
            raise ValueError("ERROR in 'ev curl': Length of the provided vector ({:d}) is smaller than number of dimensions ({:d}). Curl can't be calculated".format(numComps, numDims))

        zc0 = 0.5*(inGrid[0][0][1:] + inGrid[0][0][:-1])
        zc1 = 0.5*(inGrid[0][1][1:] + inGrid[0][1][:-1])
        outValues[..., 0] = np.gradient(inValues[0][..., 1], zc0, edge_order=2, axis=0) - np.gradient(inValues[0][..., 0], zc1, edge_order=2, axis=1)
    else:
        if numComps > 3:
            click.echo(click.style("WARNING in 'ev div': Length of the provided vector ({:d}) is longer than number of dimensions ({:d}). The last {:d} components of the vector will be disregarded.".format(numComps, numDims, numComps-numDims), fg='yellow'))
        elif numComps < 3:
            raise ValueError("ERROR in 'ev curl': Length of the provided vector ({:d}) is smaller than number of dimensions ({:d}). Curl can't be calculated".format(numComps, numDims))

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
         'sin' : { 'numIn' : 1, 'numOut' : 1, 'func' : psin },
         'cos' : { 'numIn' : 1, 'numOut' : 1, 'func' : pcos },
         'tan' : { 'numIn' : 1, 'numOut' : 1, 'func' : ptan },
         'abs' : { 'numIn' : 1, 'numOut' : 1, 'func' : absolute },
         'avg' : { 'numIn' : 2, 'numOut' : 1, 'func' : average },
         'log' : { 'numIn' : 1, 'numOut' : 1, 'func' : log },
         'log10' : { 'numIn' : 1, 'numOut' : 1, 'func' : log10 },
         'max' : { 'numIn' : 1, 'numOut' : 1, 'func' : maximum },
         'min' : { 'numIn' : 1, 'numOut' : 1, 'func' : minimum },
         'mean' : { 'numIn' : 1, 'numOut' : 1, 'func' : mean },
         'len' : { 'numIn' : 2, 'numOut' : 1, 'func' : length },
         'pow' : { 'numIn' : 2, 'numOut' : 1, 'func' : power },
         'sq' : { 'numIn' : 1, 'numOut' : 1, 'func' : sq },
         'exp' : { 'numIn' : 1, 'numOut' : 1, 'func' : exp },
         'grad' : { 'numIn' : 2, 'numOut' : 1, 'func' : grad },
         'int' : { 'numIn' : 2, 'numOut' : 1, 'func' : integrate },
         'div' : { 'numIn' : 1, 'numOut' : 1, 'func' : divergence },
         'curl' : { 'numIn' : 1, 'numOut' : 1, 'func' : curl },
}
