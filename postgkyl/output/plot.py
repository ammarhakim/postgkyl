import numpy as np
import matplotlib.pyplot as plt

def _colorbar(obj, fig, ax, label=""):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(obj, cax=cax, label=label)

def _getFig(fig=None, hold=False):
    if hold == True and fig is not None:
        return fig, ax
    else:
        return plt.subplots()

def plot(grid, values, *args, hold=False,
         streamline=False, quiver=False, contour=False):
    """Plots Gkyl data

    Unifies the plotting across a wide range of Gkyl applications. Can
    be used for both 1D an 2D data. Uses a proper colormap by default.

    Args:
    """
    if isinstance(grid, list):
        numDims = len(grid)

        idx = []
        for d in range(numDims):
            if len(grid[d]) == 1:
                idx.append(d)
        if idx:
            grid = np.delete(coords, idx)
            values = np.squeeze(values, tuple(idx)) 
            numDims = len(grid)
    else:
        numDims = 1
        grid = grid[0]

    numComps = values.shape[-1]
    if streamline or quiver:
        step = 2
    else:
        step = 1
    idxComps = range(0, numComps, step)

    fig = None
    for comp in idxComps:
        fig, ax = _getFig(fig, hold)
        
        if contour:
            im = ax.contour(grid[0], grid[1],
                            values[..., comp].transpose(), *args)
            cb = _colorbar(im, fig, ax)
        elif quiver:
            skip = int(np.max((len(grid[0]), len(grid[1])))//15)
            skip2 = int(skip//2)
            im = ax.quiver(grid[0][skip2::skip], grid[1][skip2::skip],
                           values[skip2::skip,
                                  skip2::skip,
                                  comp].transpose(),
                           values[skip2::skip,
                                  skip2::skip,
                                  comp+1].transpose())
        elif streamline:
            magnitude = np.sqrt(values[..., comp]**2 
                                + values[..., comp+1]**2)
            im = ax.streamplot(coords[0], coords[1],
                               values[..., comp].transpose(),
                               values[..., comp + 1].transpose(),
                               *args,
                               color=magnitude.transpose())
            cb = _colorbar(im.lines, fig, ax)
        else:
            if numDims == 1:
                im, = ax.plot(grid[0], values[..., comp])
            elif numDims == 2:
                im = ax.pcolormesh(grid[0], grid[1],
                                   values[..., comp].transpose())
                cb = _colorbar(im, fig, ax)
            else:
                raise ValueError("{:d}D data not yet supported".format(numDims))
    return im

   
    
    
