import click
import numpy as np
import os.path
import tkinter as tk
import bokeh.plotting as blt
from bokeh.layouts import gridplot, layout
from bokeh.models import Grid, BasicTickFormatter, ColorBar, BasicTicker, LinearColorMapper, Label
from bokeh.palettes import Inferno256
from bokeh.transform import linear_cmap


root = tk.Tk()#meassuring screen size
screen_height = root.winfo_screenheight()
root.withdraw()


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
    axLabel = ['z_0', 'z_1', 'z_2', 'z_3', 'z_4', 'z_5']
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

    sr = np.sqrt(numComps) #determine the number of rows and columns
    if sr == np.ceil(sr):
        numRows = int(sr)
        numCols = int(sr)
    elif np.ceil(sr) * np.floor(sr) >= numComps:
        numRows = int(np.floor(sr))
        numCols = int(np.ceil(sr))
    else:
        numRows = int(np.ceil(sr))
        numCols = int(np.ceil(sr))

    # Prepare the figure
    if figure is None:
        fig = []
        if numDims == 1:
            tooltips = [(axLabel[0], "$x"), ("value", "$y")] #getting tooltips ready for different dimensions
        else:
            tooltips = [(axLabel[0], "$x"), (axLabel[1], "$y"), ("value", "@image")]
        #end
        for comp in idxComps:
            fig.append(blt.figure(tooltips=tooltips,
                                  frame_height=int(screen_height*0.55/numRows),#adjust figures with the size based on the screen size
                                  frame_width=int(screen_height*0.55/numRows),
                                  outline_line_color='black',
                                  min_border_left=70,
                                  min_border_right=40)) #adjust spacings betweewn subplots to be aligned
        #end
    #end
            
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

    for comp in idxComps: 
        fig[comp].xaxis.minor_tick_line_color = None #deleting minor ticks
        fig[comp].yaxis.minor_tick_line_color = None
        #fig[comp].axis.formatter = BasicTickFormatter(precision=2) #get rid of unnecessary floating numbers.
        if numDims != 1:
            if comp % numCols != 0: #hiding labels for unnecessary subplots
                fig[comp].yaxis.major_label_text_font_size = '0pt'
            #end
            if comp < (numRows-1) * numCols:
                fig[comp].xaxis.major_label_text_font_size = '0pt'
            #end
        #end
    #end
       

    for comp in idxComps: #adding labels
        if comp >= (numRows-1) * numCols:
            fig[comp].xaxis.axis_label = axLabel[0]
        #end
        if numDims == 2:
            if comp % numCols == 0:
                fig[comp].yaxis.axis_label = axLabel[1]
            #end
        #end
    #end
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


        if numDims == 1 and label != '':
                pass
        else: 
            legend_number = Label(x=lower[0]+(upper[0]-lower[0])*0.005,
                                  y=upper[1]-(upper[1]-lower[1])*0.115, 
                                  text=label, render_mode='css',
                                  background_fill_color='white', 
                                  background_fill_alpha=0.9, 
                                  border_line_cap='round')
            fig[comp].add_layout(legend_number)
        #end
        
        # Special plots:
        if contour:
            pass
        elif streamline:
            pass
        elif quiver:
            pass
        elif diverging:
            pass

        # Basic  plots
        if numDims == 1:
            x = 0.5*(grid[0][1:]+grid[0][:-1])
            fig[comp].line(x, values[..., comp], line_width=2, legend=label)
        else:
        
            fig[comp].image(image=[values[..., comp].transpose()],
                      x=lower[0], y=lower[1],
                      dw=(upper[0]-lower[0]), dh=(upper[1]-lower[1]),
                      palette="Inferno256")
            mapper = LinearColorMapper(palette='Inferno256',
                                    low=np.amin(values[...,comp]), 
                                    high=np.amax(values[...,comp]))#adding a color bar
            color_bar = ColorBar(color_mapper=mapper, 
                                 width=7, 
                                 location=(0,0), 
                                 formatter=BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                 ticker=BasicTicker(desired_num_ticks=4), 
                                 label_standoff=10, 
                                 border_line_color=None,
                                 padding=2,
                                 bar_line_color='black')
            fig[comp].add_layout(color_bar, 'right')
        #end

        #-------------------------------------------------------------
        #-- Additional Formatting ------------------------------------
        for comp in idxComps:
            fig[comp].x_range.range_padding  = 0
            if numDims == 2:
                fig[comp].y_range.range_padding = 0
            #end
        #end

        #mapper = linear_cmap(palette=Inferno256)
        #color_bar = ColorBar (color_mapper=mapper['transform'], width=0.7)   
        #for comp in idxComps:
            #fig[comp].add_layout(color_bar, 'right')
        #fig.x_range.range_padding  = 0
        #if numDims == 2:
        #    fig.y_range.range_padding = 0

        #end

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

    gp = gridplot(children=fig, toolbar_location='right', ncols=numCols, merge_tools=True)
    #blt.output_file("image.html", title="Postgkyl output")
    #blt.show(gp)
    return gp
#end
