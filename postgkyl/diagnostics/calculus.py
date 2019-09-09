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
            #end
        else:
            raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")
        #end
    else:
        numDims = data.getNumDims()
        axis = tuple(range(numDims))
    #end

    # Get dz elements
    dz = []
    for d, coord in enumerate(grid):
        dz.append(coord[1:] - coord[:-1])
        if len(coord) > 1 and len(coord) == values.shape[d]:
            dz[-1] = np.append(dz[-1], dz[-1][-1])
        #end
    #end

    # Integration assuming values are cell centered averages
    # Should work for nonuniform meshes
    for ax in sorted(axis, reverse=True):
        if len(grid[ax]) > 1:
            values = np.moveaxis(values, ax, -1)
            values = np.dot(values, dz[ax])
        else:
            values = values.mean(axis=ax)
        #end
    #end

    for ax in sorted(axis):
        grid[ax] = np.array([grid[ax].mean()])
        values = np.expand_dims(values, ax)
    #end

    if stack is False:
        return grid, values
    else:
        data.pushGrid(grid)
        data.pushValues(values)
    #end


def grad(data):
    pass

def div(data):
    pass

def curl(data):
    pass
