import numpy as np

class GCalculus(object):
    """Postgkyl class for integration and differential operators.
    """

    def __init__(self, data):
        self.data = data

    def integrate(self, axis, stack=False):
        grid = list(self.data.peakGrid())
        values = np.copy(self.data.peakValues())

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
                    axis = np.zeros(bounds[1]-bounds[0], np.int)
                    axis += int(bounds[0])
                    axis = tuple(axis)
                else:
                    axis = tuple([int(axis)])
            else:
                raise TypeError("'axis' needs to be integer, tuple, string of comma separated integers, or a slice ('int:int')")
        else:
            numDims = self.data.getNumDims()
            axis = tuple(range(numDims))

        # Get dz elements
        dz = []
        for coord in grid:
            dz.append(coord - np.roll(coord, 1))
            dz[-1][0] = dz[-1][1]  # "Fix" the first element

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
            self.data.pushGrid(grid)
            self.data.pushValues(values)

    def grad(self):
        pass

    def div(self):
        pass

    def curl(self):
        pass
