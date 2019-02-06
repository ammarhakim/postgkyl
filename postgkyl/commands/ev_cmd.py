import numpy as np

def add(inGrid, inValues):
    if inGrid[0] is not []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[0] + inValues[1]
    return outGrid, outValues

def subtract(inGrid, inValues):
    if inGrid[0] is not []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] - inValues[0]
    return outGrid, outValues

def mult(inGrid, inValues):
    if inGrid[0] is not []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] * inValues[0]
    return outGrid, outValues

def divide(inGrid, inValues):
    if inGrid[0] is not []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = inValues[1] / inValues[0]
    return outGrid, outValues

def sqrt(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.sqrt(inValues[0])
    return outGrid, outValues

def absolute(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.abs(inValues[0])
    return outGrid, outValues

def log(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.log(inValues[0])
    return outGrid, outValues

def log10(inGrid, inValues):
    outGrid = inGrid[0]
    outValues = np.log10(inValues[0])
    return outGrid, outValues 

def minimum(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.min(inValues[0]))
    return outGrid, outValues 

def maximum(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.max(inValues[0]))
    return outGrid, outValues 

def mean(inGrid, inValues):
    outGrid = []
    outValues = np.atleast_1d(np.mean(inValues[0]))
    return outGrid, outValues 

def power(inGrid, inValues):
    if inGrid[0] is not []:
        outGrid = inGrid[0]
    else:
        outGrid = inGrid[1]
    outValues = np.pow(inValues[1], inValues[0])
    return outGrid, outValues 

def grad(inGrid, inValues):
    outGrid = inGrid[1]
    axis = int(inValues[0])
    zc = 0.5*(inGrid[1][axis][1:] + inGrid[1][axis][:-1]) # get cell centered values
    outValues = np.gradient(inValues[1], zc, edge_order=2, axis=axis)
    return outGrid, outValues 

cmds = { '+' : { 'numIn' : 2, 'func' : add }, 
         '-' : { 'numIn' : 2, 'func' : subtract },
         '*' : { 'numIn' : 2, 'func' : mult },
         '/' : { 'numIn' : 2, 'func' : divide },
         'sqrt' : { 'numIn' : 1, 'func' : sqrt },
         'abs' : { 'numIn' : 1, 'func' : absolute },
         'log' : { 'numIn' : 1, 'func' : log },
         'log10' : { 'numIn' : 1, 'func' : log10 },
         'max' : { 'numIn' : 1, 'func' : maximum },
         'min' : { 'numIn' : 1, 'func' : minimum },
         'mean' : { 'numIn' : 1, 'func' : mean },
         'pow' : { 'numIn' : 2, 'func' : power },
         'grad' : { 'numIn' : 2, 'func' : grad },
}
