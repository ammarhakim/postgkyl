import numpy as np

def integrate(data, axis, stack=False):
    grid = list(data.getGrid())
    values = np.copy(data.getValues())

    # Convert Python input to an input Numpy understands
    if axis is not None:
        if isinstance(axis, int):
            axis = tuple([axis])
        elif isinstance(axis, tuple):
            pass
        elif isinstance(axis, str):
            if len(axis.split(',')) > 1:
                axes = axis.split(',')
                axis = tuple([int(a) for a in axes])
            elif len(axis.split(':')) == 2:
                bounds = axis.split(':')
                #axis = np.zeros(bounds[1]-bounds[0], np.int)
                #axis += int(bounds[0])
                axis = tuple(range(bouns[0], bounds[1]))
            else:
                axis = tuple([int(axis)])
        else:
            raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")
    else:
        numDims = data.getNumDims()
        axis = tuple(range(numDims))

    # Get dz elements
    dz = []
    for coord in grid:
        dz.append(coord[1:] - coord[:-1])

    # Integration assuming values are cell centered averages
    # Should work for nonuniform meshes
    for ax in sorted(axis, reverse=True):
        values = np.moveaxis(values, ax, -1)
        values = np.dot(values, dz[ax])

    for ax in sorted(axis):
        grid[ax] = np.array([0])
        values = np.expand_dims(values, ax)

    if stack is False:
        return grid, values
    else:
        data.pushGrid(grid)
        data.pushValues(values)

def grad(data):
    pass

def div(data):
    pass

def curl(data):
    pass
