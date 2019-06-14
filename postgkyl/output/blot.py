import click
import numpy as np
import os.path
import bokeh.plotting as blt

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


def blot(gdata, args=(),
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
    # if style is None:
    #     plt.style.use(os.path.dirname(os.path.realpath(__file__)) \
    #                   + "/postgkyl.mplstyle")
    # else:
    #     plt.style.use(style)
    #end

    #-----------------------------------------------------------------
    #-- Data Loading -------------------------------------------------
    numDims = gdata.getNumDims(squeeze=True)
    if numDims > 2:
        raise Exception('Only 1D and 2D plots are currently supported')
    #end    
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
            #end
        #end
        if idx:
            grid = np.delete(grid, idx)
            lower = np.delete(lower, idx)
            upper = np.delete(upper, idx)
            cells = np.delete(cells, idx)
            axLabel = np.delete(axLabel, idx)
            values = np.squeeze(values, tuple(idx)) 
        #end
    numComps = values.shape[-1]
    if streamline or quiver:
        step = 2
    else:
        step = 1
    #end
    idxComps = range(int(np.floor(numComps/step)))
    numComps = len(idxComps)

    if xscale != 1.0:
        axLabel[0] = axLabel[0] + r' $\times$ {:.3e}'.format(xscale)
    #end
    if numDims == 2 and yscale != 1.0:
        axLabel[1] = axLabel[1] + r' $\times$ {:.3e}'.format(yscale)
    #end

    # Prepare the figure
    if figure is None:
        fig = blt.figure()
        fig.x_range.range_padding = fig.y_range.range_padding = 0
    # elif isinstance(figure, int):
    #     fig = plt.figure(figure)
    # elif isinstance(figure, matplotlib.figure.Figure):
    #     fig = figure
    # elif isinstance(figure, str) or isinstance(figure, unicode):
    #     fig = plt.figure(int(figure))
    # else:
    #     raise TypeError(("'fig' keyword needs to be one of "
    #                      "None (default), int, or MPL Figure"))
    #end

    #-----------------------------------------------------------------
    #-- Preparing the Axes -------------------------------------------
    # if fig.axes:
    #     ax = fig.axes
    #     if squeeze is False and numComps > len(ax):
    #         raise ValueError(
    #             "Trying to plot into figure with not enough axes")
    #     #end
    # else:
    #     if squeeze:  # Plotting into 1 panel
    #         plt.subplots(1, 1, num=fig.number)
    #         ax = fig.axes
    #         if xlabel is None:
    #             ax[0].set_xlabel(axLabel[0])
    #             if group == 1:
    #                 ax[0].set_xlabel(axLabel[1])
    #             #end
    #         else:
    #             ax[0].set_xlabel(xlabel)
    #         #end
    #         if ylabel is None:
    #             if numDims == 2 and group is None:
    #                 ax[0].set_ylabel(axLabel[1])
    #             #end
    #         else:
    #             ax[0].set_ylabel(ylabel)
    #         #end
    #         if title is not None:
    #             ax[0].set_title(title, y=1.08)
    #         #end
    #     else:  # Plotting each components into its own subplot
    #         sr = np.sqrt(numComps)
    #         if sr == np.ceil(sr):
    #             numRows = int(sr)
    #             numCols = int(sr)
    #         elif np.ceil(sr) * np.floor(sr) >= numComps:
    #             numRows = int(np.floor(sr))
    #             numCols = int(np.ceil(sr))
    #         else:
    #             numRows = int(np.ceil(sr))
    #             numCols = int(np.ceil(sr))
    #         #end

    #         if numDims == 1 or group is not None: 
    #             plt.subplots(numRows, numCols,
    #                          sharex=True,
    #                          num=fig.number)
    #         else: # In 2D, share y-axis as well
    #             plt.subplots(numRows, numCols,
    #                          sharex=True, sharey=True,
    #                          num=fig.number)
    #         #end
    #         ax = fig.axes
    #         # Adding labels only to the right subplots
    #         for comp in idxComps:
    #             if comp >= (numRows-1) * numCols:
    #                 if xlabel is None:
    #                     ax[comp].set_xlabel(axLabel[0])
    #                     if group == 1:
    #                         ax[comp].set_xlabel(axLabel[1])
    #                     #end
    #                 else:
    #                     ax[comp].set_xlabel(xlabel)
    #                 #end
    #             #end
    #             if comp % numCols == 0:
    #                 if ylabel is None:
    #                     if numDims == 2 and group is None:
    #                         ax[comp].set_ylabel(axLabel[1])
    #                     #end
    #                 else:
    #                     ax[comp].set_ylabel(ylabel)
    #                 #end
    #             #end
    #             if comp < numCols and title is not None:
    #                 ax[comp].set_title(title, y=1.08)
    #             #end
    #         #end
    #     #end
    # #end

    #-----------------------------------------------------------------
    #-- Main Plotting Loop -------------------------------------------
    for comp in idxComps:

        if len(idxComps) > 1:
            if labelPrefix == "":
                label = str(comp)
            else:
                label = '{:s}_c{:d}'.format(labelPrefix, comp)
            #end
        else:
            label = labelPrefix
        #end
        # Special plots:
        fig.image(image=[values[..., comp].transpose()], x=lower[0], y=lower[1], dw=(upper[0]-lower[0]), dh=(upper[1]-lower[1]), palette="Inferno11")
        #end

        #-------------------------------------------------------------
        #-- Additional Formatting ------------------------------------
        # cax.grid(True)
        # # Legend
        # if legend:
        #     if numDims == 1 and label != '':
        #         cax.legend(loc=0)
        #     else:
        #         cax.text(0.03, 0.96, label,
        #                  bbox=dict(facecolor='w', edgecolor='w', alpha=0.8,
        #                            boxstyle="round"),
        #                  verticalalignment='top',
        #                  horizontalalignment='left',
        #                  transform=cax.transAxes)
        #     #end
        # #end
        # if logx:
        #     cax.set_xscale('log')
        # #end
        # if logy:
        #     cax.set_yscale('log')
        # #end
        # if numDims == 1:
        #     if vmin is not None and vmax is not None:
        #         cax.set_ylim(vmin, vmax)
        #     #end
        #     plt.autoscale(enable=True, axis='x', tight=True)
        # elif numDims == 2:
        #     if fixaspect:
        #         plt.setp(cax, aspect=1.0)
        #     #end
        # #end
    #end
    #for i in range(numComps, len(ax)):
    #    ax[i].axis('off')
    ##end
    
    blt.output_file("image.html", title="image.py example")
    blt.show(fig)
    return fig
#end
