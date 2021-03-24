import click
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os.path

# Helper functions
def _colorbar(obj, fig, cax, label="", extend=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(cax)
    cax2 = divider.append_axes("right", size="3%", pad=0.05)
    if extend is not None:
        return fig.colorbar(obj, cax=cax2, label=label or "", extend=extend)
    else:
        return fig.colorbar(obj, cax=cax2, label=label or "")
    #end
#end

def _gridNodalToCellCentered(grid, cells):
    numDims = len(grid)
    gridOut = []
    if numDims != len(cells):  # sanity check
        raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
    #end
    for d in range(numDims):
        if len(grid[d].shape) == 1:
            if grid[d].shape[0] == cells[d]:
                gridOut.append(grid[d])
            elif grid[d].shape[0] == cells[d]+1:
                gridOut.append(0.5*(grid[d][:-1]+grid[d][1:]))
            else:
                raise ValueError("Something is terribly wrong...")
            #end
        else:
            pass
        #end
    #end
    return gridOut
#end


def plot(data, args=(),
         figure=None, squeeze=False,
         numAxes=None, startAxes=0,
         nSubplotRow=None, nSubplotCol=None,
         scatter=False,
         streamline=False, quiver=False, contour=False,
         clevels=None,
         diverging=False, group=None,
         xscale=1.0, yscale=1.0,
         style=None, legend=True, labelPrefix='',
         xlabel=None, ylabel=None, clabel=None, title=None,
         logx=False, logy=False, logz=False,
         fixaspect=False,
         vmin=None, vmax=None, edgecolors=None,
         xlim=None, ylim=None,
         showgrid=True,
         hashtag=False, xkcd=False,
         color=None, markersize=None,
         transpose=False,
         figsize=None,
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
    #end

    # The most important thing
    if xkcd:
        plt.xkcd()
    #end
    
    #-----------------------------------------------------------------
    #-- Data Loading -------------------------------------------------
    numDims = data.getNumDims(squeeze=True)
    if numDims > 2:
        raise Exception('Only 1D and 2D plots are currently supported')
    #end    
    # Get the handles on the grid and values
    grid = data.getGrid().copy()
    values = data.getValues()
    lower, upper = data.getBounds()
    cells = data.getNumCells()
    # Squeeze the data (get rid of "collapsed" dimensions)
    axLabel = ['$z_0$', '$z_1$', '$z_2$', '$z_3$', '$z_4$', '$z_5$']
    if len(grid) > numDims:
        idx = []
        for d in range(len(grid)):
            if cells[d] <= 1:
                idx.append(d)
            #end
        #end
        if idx:
            for i in reversed(idx):
                grid.pop(i)
            #end
            lower = np.delete(lower, idx)
            upper = np.delete(upper, idx)
            cells = np.delete(cells, idx)
            axLabel = np.delete(axLabel, idx)
            values = np.squeeze(values, tuple(idx))
        #end
    #end
    if streamline or quiver:
        step = 2
    else:
        step = 1
    #end
    numComps = values.shape[-1]
    idxComps = range(int(np.floor(numComps/step)))
    if numAxes:
        numComps = numAxes
    else:
        numComps = len(idxComps)
    #end

    if xscale != 1.0:
        axLabel[0] = axLabel[0] + r' $\times$ {:.3e}'.format(xscale)
    #end
    if numDims == 2 and yscale != 1.0:
        axLabel[1] = axLabel[1] + r' $\times$ {:.3e}'.format(yscale)
    #end

    # Prepare the figure
    if figsize:
        figsize = (int(figsize.split(',')[0]),
                   int(figsize.split(',')[1]))
    #end
    if figure is None:
        fig = plt.figure(figsize=figsize)
    elif isinstance(figure, int):
        fig = plt.figure(figure, figsize=figsize)
    elif isinstance(figure, matplotlib.figure.Figure):
        fig = figure
    elif isinstance(figure, str) or isinstance(figure, unicode):
        fig = plt.figure(int(figure))
    else:
        raise TypeError(("'fig' keyword needs to be one of "
                         "None (default), int, or MPL Figure"))
    #end

    #-----------------------------------------------------------------
    #-- Preparing/loading Axes ---------------------------------------
    if fig.axes:
        ax = fig.axes
        if squeeze is False and numComps > len(ax):
            raise ValueError(
                "Trying to plot into figure with not enough axes")
        #end
    else:
        if squeeze:  # Plotting into 1 panel
            plt.subplots(1, 1, num=fig.number)
            ax = fig.axes
            if xlabel is None:
                ax[0].set_xlabel(axLabel[0])
                if group == 1:
                    ax[0].set_xlabel(axLabel[1])
                #end
            else:
                ax[0].set_xlabel(xlabel)
            #end
            if ylabel is None:
                if numDims == 2 and group is None:
                    ax[0].set_ylabel(axLabel[1])
                #end
            else:
                ax[0].set_ylabel(ylabel)
            #end
            if title is not None:
                ax[0].set_title(title, y=1.08)
            #end
        else:  # Plotting each components into its own subplot
            if nSubplotRow is not None:
                numRows = nSubplotRow
                numCols = int(np.ceil(numComps/numRows))
            elif nSubplotCol is not None:
                numCols = nSubplotCol
                numRows = int(np.ceil(numComps/numCols))
            else:
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
                #end
            #end

            if numDims == 1 or group is not None: 
                plt.subplots(numRows, numCols,
                             sharex=True,
                             num=fig.number)
            else: # In 2D, share y-axis as well
                plt.subplots(numRows, numCols,
                             sharex=True, sharey=True,
                             num=fig.number)
            #end
            ax = fig.axes
            # Removing extra axes
            for i in range(numComps, len(ax)):
                ax[i].axis('off')
            #end
            # Adding labels only to the right subplots
            for axIdx in range(len(ax)):
                if axIdx >= (numRows-1) * numCols:
                    if xlabel is None:
                        ax[axIdx].set_xlabel(axLabel[0])
                        if group == 1:
                            ax[axIdx].set_xlabel(axLabel[1])
                        #end
                    else:
                        ax[axIdx].set_xlabel(xlabel)
                    #end
                #end
                if axIdx % numCols == 0:
                    if ylabel is None:
                        if numDims == 2 and group is None:
                            ax[axIdx].set_ylabel(axLabel[1])
                        #end
                    else:
                        ax[axIdx].set_ylabel(ylabel)
                    #end
                #end
                if axIdx < numCols and title is not None:
                    ax[axIdx].set_title(title, y=1.08)
                #end
            #end
        #end
    #end

    #-----------------------------------------------------------------
    #-- Main Plotting Loop -------------------------------------------
    for comp in idxComps:
        if squeeze:
            cax = ax[0]
        else:
            cax = ax[comp+startAxes]
        #end
        if len(idxComps) > 1:
            if labelPrefix == "":
                label = str(comp)
            else:
                label = '{:s}_c{:d}'.format(labelPrefix, comp)
            #end
        else:
            label = labelPrefix
        #end
        if numDims == 1:
            cl = data.color
            if color is not None:
                cl = color
            #end
            gridCC = _gridNodalToCellCentered(grid, cells)
            im = cax.plot(gridCC[0]*xscale,
                          values[..., comp],
                          *args, label=label,
                          color=cl, markersize=markersize)
            xmin = min(gridCC[0]*xscale)
            xmax = max(gridCC[0]*xscale)
        elif numDims == 2:            
            if contour:  #--------------------------------------------
                levels = 10
                if clevels:
                    if ":" in clevels:
                        s = clevels.split(":")
                        levels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
                    else:
                        levels = int(clevels)
                        
                
                gridCC = _gridNodalToCellCentered(grid, cells)
                im = cax.contour(gridCC[0]*xscale, gridCC[1]*yscale,
                                 values[..., comp].transpose(), levels,
                                 *args)
                cb = _colorbar(im, fig, cax, label=clabel)
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
                _colorbar(im.lines, fig, cax, label=clabel)
            elif diverging:  #----------------------------------------
                vmax = np.abs(values[..., comp]).max()
                im = cax.pcolormesh(grid[0]*xscale, grid[1]*yscale,
                                    values[..., comp].transpose(),
                                    vmax=vmax, vmin=-vmax,
                                    cmap='RdBu_r',
                                    edgecolors=edgecolors, linewidth=0.1,
                                    *args)
                _colorbar(im, fig, cax, label=clabel)
            elif group is not None:  #--------------------------------
                if group == 0:
                    numLines = values.shape[1]
                else:
                    numLines = values.shape[0]
                #end
                gridCC = _gridNodalToCellCentered(grid, cells)
                for l in range(numLines):
                    idx = [slice(0, u) for u in values.shape]
                    idx[-1] = comp
                    color = cm.viridis(l / (numLines-1))
                    if group == 0:
                        idx[1] = l
                        im = cax.plot(gridCC[0]*xscale, values[tuple(idx)],
                                      *args, color=color)
                        mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=gridCC[1][0]*yscale,vmax=gridCC[1][-1]*yscale,clip=False), cmap=cm.viridis)
                        label = clabel or 'Z1' 
                    else:
                        idx[0] = l
                        im = cax.plot(gridCC[1]*yscale, values[tuple(idx)],
                                      *args, color=color)
                        mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=gridCC[0][0]*xscale,vmax=gridCC[0][-1]*xscale,clip=False), cmap=cm.viridis)
                        label = clabel or 'Z0' 
                fig.colorbar(mappable, ax=cax, label=label)
                legend = False
            else:
                extend = None
                if vmin is not None and vmax is not None:
                    extend = 'both'
                elif vmax is not None:
                    extend = 'max'
                elif vmin is not None:
                    extend = 'min'
                #end
                if logz:
                    tmp = np.array(values[..., comp])
                    if vmin is not None or vmax is not None:
                        for i in range(tmp.shape[0]):
                            for j in range(tmp.shape[1]):
                                if vmin and tmp[i, j] < vmin:
                                    tmp[i, j] = vmin
                                #end
                                if vmax and tmp[i, j] > vmax:
                                    tmp[i, j] = vmax
                                #end
                            #end
                        #end
                    #end
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                    im = cax.pcolormesh(grid[0]*xscale, grid[1]*yscale,
                                        tmp.transpose(),
                                        norm=norm,# vmin=vmin, vmax=vmax,
                                        edgecolors=edgecolors,
                                        linewidth=0.1,
                                        *args)
                elif transpose:
                    im = cax.pcolormesh(grid[1]*xscale, grid[0]*yscale,
                                        values[..., comp],
                                        vmin=vmin, vmax=vmax,
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
                #end
                _colorbar(im, fig, cax, extend=extend, label=clabel)
            #end
        else:
            raise ValueError("{:d}D data not yet supported".
                             format(numDims))
        #end
        #-------------------------------------------------------------
        #-- Additional Formatting ------------------------------------
        cax.grid(showgrid)
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
            #end
        #end
        if hashtag:
            cax.text(0.97, 0.03, '#pgkyl',
                     bbox=dict(facecolor='w', edgecolor='w', alpha=0.8,
                               boxstyle="round"),
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     transform=cax.transAxes)
        #end
        if logx:
            cax.set_xscale('log')
        #end
        if logy:
            cax.set_yscale('log')
        #end

        if xlim is not None:
            limSplit = xlim.split(',')
            cax.set_xlim(float(limSplit[0]), float(limSplit[1]))
        #end
        if ylim is not None:
            limSplit = ylim.split(',')
            cax.set_ylim(float(limSplit[0]), float(limSplit[1]))
        #end
        if numDims == 1:
            if vmin is not None and vmax is not None:
                cax.set_ylim(vmin, vmax)
            #end
            if xlim is None:
                cax.set_xlim(xmin, xmax)
            #end
        elif numDims == 2:
            if fixaspect:
                plt.setp(cax, aspect=1.0)
            #end
        #end
        if xlim is None:
            plt.autoscale(enable=True, axis='x', tight=True)
        #end
    #end
    plt.tight_layout()
    return im
#end
