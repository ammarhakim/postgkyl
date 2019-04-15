import click
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import os.path

# Helper functions
def _colorbar(obj, fig, ax, label="", extend=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    if extend is not None:
        return fig.colorbar(obj, cax=cax, label=label, extend=extend)
    else:
        return fig.colorbar(obj, cax=cax, label=label)

def _gridNodalToCellCentered(grid, cells):
    numDims = len(grid)
    gridOut = []
    if numDims != len(cells):  # sanity check
        raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
    for d in range(numDims):
        if len(grid[d].shape) == 1:
            if grid[d].shape[0] == cells[d]:
                gridOut.append(grid[d])
            elif grid[d].shape[0] == cells[d]+1:
                gridOut.append(0.5*(grid[d][:-1]+grid[d][1:]))
            else:
                raise ValueError("Something is terribly wrong...")
        else:
            pass
    return gridOut
        


def plot(gdata, args=(),
         figure=None, squeeze=False,
         streamline=False, quiver=False, contour=False,
         diverging=False, group=None,
         xscale=1.0, yscale=1.0,
         style=None, legend=True, labelPrefix='',
         xlabel=None, ylabel=None, title=None,
         logx=False, logy=False, logz=False,
         color=None, fixaspect=False,
         vmin=None, vmax=None, edgecolors=None,
         **kwargs):
    """Plots Gkeyll data

    Unifies the plotting across a wide range of Gkyl applications. Can
    be used for both 1D an 2D data. Uses a proper colormap by default.

    Args:
    """
    # Load Postgkyl style
    if style is None:
        plt.style.use(os.path.dirname(os.path.realpath(__file__)) \
                      + "/postgkyl.mplstyle")
    else:
        plt.style.use(style)

    #-----------------------------------------------------------------
    #-- Data Loading -------------------------------------------------
    numDims = gdata.getNumDims(squeeze=True)
    if numDims > 2:
        raise Exception('Only 1D and 2D plots are currently supported')
        
    # Get the handles on the grid and values
    grid = gdata.getGrid()
    values = gdata.getValues()
    lower, upper = gdata.getBounds()
    cells = gdata.getNumCells()
    # Squeeze the data (get rid of "collapsed" dimensions)
    axLabel = ['$z_0$', '$z_1$', '$z_2$', '$z_3$', '$z_4$', '$z_5$']
    if len(grid) > numDims:
        idx = []
        for d in range(len(grid)):
            if cells[d] <= 1:
                idx.append(d)
        if idx:
            grid = np.delete(grid, idx)
            lower = np.delete(lower, idx)
            upper = np.delete(upper, idx)
            cells = np.delete(cells, idx)
            axLabel = np.delete(axLabel, idx)
            values = np.squeeze(values, tuple(idx)) 

    numComps = values.shape[-1]
    if streamline or quiver:
        step = 2
    else:
        step = 1
    idxComps = range(int(np.floor(numComps/step)))
    numComps = len(idxComps)

    if xscale != 1.0:
        axLabel[0] = axLabel[0] + r' $\times$ {:.3e}'.format(xscale)
    if numDims == 2 and yscale != 1.0:
        axLabel[1] = axLabel[1] + r' $\times$ {:.3e}'.format(yscale)

    # Prepare the figure
    if figure is None:
        fig = plt.figure()
    elif isinstance(figure, int):
        fig = plt.figure(figure)
    elif isinstance(figure, matplotlib.figure.Figure):
        fig = figure
    elif isinstance(figure, str) or isinstance(figure, unicode):
        fig = plt.figure(int(figure))
    else:
        raise TypeError(("'fig' keyword needs to be one of "
                         "None (default), int, or MPL Figure"))

    #-----------------------------------------------------------------
    #-- Preparing the Axes -------------------------------------------
    if fig.axes:
        ax = fig.axes
        if squeeze is False and numComps > len(ax):
            raise ValueError(
                "Trying to plot into figure with not enough axes")
    else:
        if squeeze:  # Plotting into 1 panel
            plt.subplots(1, 1, num=fig.number)
            ax = fig.axes
            if xlabel is None:
                ax[0].set_xlabel(axLabel[0])
                if group == 1:
                    ax[0].set_xlabel(axLabel[1])
            else:
                ax[0].set_xlabel(xlabel)
            if ylabel is None:
                if numDims == 2 and group is None:
                    ax[0].set_ylabel(axLabel[1])
            else:
                ax[0].set_ylabel(ylabel)
            if title is not None:
                ax[0].set_title(title, y=1.08)
        else:  # Plotting each components into its own subplot
            sr = np.sqrt(numComps)
            if sr == np.ceil(sr):
                numRows = int(sr)
                numCols = int(sr)
            elif np.ceil(sr) * np.floor(sr) >= numComps:
                numRows = int(np.floor(sr))
                numCols = int(np.ceil(sr))
            else:
                numRows = int(np.ceil(sr))
                numCols = int(np.ceil(sr))

            if numDims == 1 or group is not None: 
                plt.subplots(numRows, numCols,
                             sharex=True,
                             num=fig.number)
            else: # In 2D, share y-axis as well
                plt.subplots(numRows, numCols,
                             sharex=True, sharey=True,
                             num=fig.number)
            ax = fig.axes
            # Adding labels only to the right subplots
            for comp in idxComps:
                if comp >= (numRows-1) * numCols:
                    if xlabel is None:
                        ax[comp].set_xlabel(axLabel[0])
                        if group == 1:
                            ax[comp].set_xlabel(axLabel[1])
                    else:
                        ax[comp].set_xlabel(xlabel)
                if comp % numCols == 0:
                    if ylabel is None:
                        if numDims == 2 and group is None:
                            ax[comp].set_ylabel(axLabel[1])
                    else:
                        ax[comp].set_ylabel(ylabel)
                if comp < numCols and title is not None:
                    ax[comp].set_title(title, y=1.08)


    #-----------------------------------------------------------------
    #-- Main Plotting Loop -------------------------------------------
    for comp in idxComps:
        if squeeze:
            cax = ax[0]
        else:
            cax = ax[comp]
        if len(idxComps) > 1:
            label = '{:s}c{:d}'.format(labelPrefix, comp)
        else:
            label = labelPrefix
            
        # Special plots:
        if numDims == 1:
            gridCC = _gridNodalToCellCentered(grid, cells)
            im = cax.plot(gridCC[0]*xscale,
                          values[..., comp],
                          *args, label=label)
        elif numDims == 2: 
            if contour:  #--------------------------------------------
                gridCC = _gridNodalToCellCentered(grid, cells)
                im = cax.contour(gridCC[0]*xscale, gridCC[1]*yscale,
                                 values[..., comp].transpose(),
                                 *args)
                cb = _colorbar(im, fig, cax)
            elif quiver:  #-------------------------------------------
                skip = int(np.max((len(grid[0]), len(grid[1])))//15)
                skip2 = int(skip//2)
                gridCC = _gridNodalToCellCentered(grid, cells)
                im = cax.quiver(gridCC[0][skip2::skip]*xscale,
                                gridCC[1][skip2::skip]*yscale,
                                values[skip2::skip,
                                       skip2::skip,
                                       2*comp].transpose(),
                                values[skip2::skip,
                                       skip2::skip,
                                       2*comp+1].transpose())
            elif streamline:  #---------------------------------------
                magnitude = np.sqrt(values[..., 2*comp]**2 
                                    + values[..., 2*comp+1]**2)
                gridCC = _gridNodalToCellCentered(grid, cells)
                im = cax.streamplot(gridCC[0]*xscale, gridCC[1]*yscale,
                                    values[..., 2*comp].transpose(),
                                    values[..., 2*comp+1].transpose(),
                                    *args,
                                    color=magnitude.transpose())
                cb = _colorbar(im.lines, fig, cax)
            elif diverging:  #----------------------------------------
                vmax = np.abs(values[..., comp]).max()
                im = cax.pcolormesh(grid[0]*xscale, grid[1]*yscale,
                                    values[..., comp].transpose(),
                                    vmax=vmax, vmin=-vmax,
                                    cmap='RdBu_r',
                                    edgecolors=edgecolors, linewidth=0.1,
                                    *args)
                cb = _colorbar(im, fig, cax)
            elif group is not None:  #--------------------------------
                if group == 0:
                    numLines = values.shape[1]
                else:
                    numLines = values.shape[0]
                gridCC = _gridNodalToCellCentered(grid, cells)
                for l in range(numLines):
                    idx = [slice(0, u) for u in values.shape]
                    idx[-1] = comp
                    color = cm.inferno(l / (numLines-1))
                    if group == 0:
                        idx[1] = l
                        im = cax.plot(gridCC[0]*xscale, values[tuple(idx)],
                                      *args, color=color)
                    else:
                        idx[0] = l
                        im = cax.plot(gridCC[1]*yscale, values[tuple(idx)],
                                      *args, color=color)
                legend = False
            else:  # Basic plots -------------------------------------
                extend = None
                if vmin is not None and vmax is not None:
                    extend = 'both'
                elif vmax is not None:
                    extend = 'max'
                elif vmin is not None:
                    extend = 'min'

                if logz:
                    tmp = np.array(values[..., comp])
                    if vmin is not None or vmax is not None:
                        for i in range(tmp.shape[0]):
                            for j in range(tmp.shape[1]):
                                if vmin and tmp[i, j] < vmin:
                                    tmp[i, j] = vmin
                                if vmax and tmp[i, j] > vmax:
                                    tmp[i, j] = vmax

                    norm = cl.LogNorm(vmin=vmin, vmax=vmax)
                    im = cax.pcolormesh(grid[0]*xscale, grid[1]*yscale,
                                        tmp.transpose(),
                                        norm=norm,# vmin=vmin, vmax=vmax,
                                        edgecolors=edgecolors,
                                        linewidth=0.1,
                                        *args)
                else:
                    im = cax.pcolormesh(grid[0]*xscale, grid[1]*yscale,
                                        values[..., comp].transpose(),
                                        vmin=vmin, vmax=vmax,
                                        edgecolors=edgecolors,
                                        linewidth=0.1,
                                        *args) 
                cb = _colorbar(im, fig, cax, extend=extend)
        else:
            raise ValueError("{:d}D data not yet supported".
                             format(numDims))


        #-------------------------------------------------------------
        #-- Additional Formatting ------------------------------------
        cax.grid(True)
        # Legend
        if legend:
            if numDims == 1 and label != '':
                cax.legend(loc=0)
            else:
                cax.text(0.03, 0.96, label,
                         bbox=dict(facecolor='w', edgecolor='w', alpha=0.8,
                                   boxstyle="round"),
                         verticalalignment='top',
                         horizontalalignment='left',
                         transform=cax.transAxes)

        if logx:
            cax.set_xscale('log')
        if logy:
            cax.set_yscale('log')

        if numDims == 1:
            if vmin is not None and vmax is not None:
                cax.set_ylim(vmin, vmax)
            plt.autoscale(enable=True, axis='x', tight=True)
        elif numDims == 2:
            if fixaspect:
                plt.setp(cax, aspect=1.0)

    for i in range(numComps, len(ax)):
        ax[i].axis('off')

    plt.tight_layout()
    return im
