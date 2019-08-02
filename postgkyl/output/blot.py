from matplotlib import cm
from postgkyl.utils import streamlines
import click
import numpy as np
import os.path
import bokeh.plotting as blt
import bokeh.models as bm
import bokeh.layouts as bl
import bokeh.transform as bt
import bokeh.colors as bc
import bokeh.palettes
import matplotlib.pyplot as plt

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
            if streamline or quiver:
                tooltips = []
            else:
                tooltips = [(axLabel[0], "$x"), (axLabel[1], "$y"), ("value", "@image")]
            #end
        #end                               
        for comp in idxComps:
            if logx:
                fig.append(blt.figure(tooltips=tooltips,
                                    x_axis_type = "log",
                                    frame_height=int(600.0/numRows),#adjust figures with the size based on the screen size
                                    frame_width=int(600.0/numRows),
                                    outline_line_color='black',
                                    min_border_left=70,
                                    min_border_right=40)) #adjust spacings betweewn subplots to be aligned
            elif logy:
                    fig.append(blt.figure(tooltips=tooltips,
                                          y_axis_type = "log",
                                          frame_height=int(600.0/numRows),#adjust figures with the size based on the screen size
                                          frame_width=int(600.0/numRows),
                                          outline_line_color='black',
                                          min_border_left=70,
                                          min_border_right=40)) #adjust spacings betweewn subplots to be aligned
            elif logx and logy:
                pass
            else:
                fig.append(blt.figure(tooltips=tooltips,
                                    frame_height=int(600.0/numRows),#adjust figures with the size based on the screen size
                                    frame_width=int(600.0/numRows),
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
        fig[comp].xaxis.major_label_text_font_size = '12pt' #tick font size adjustment
        fig[comp].yaxis.major_label_text_font_size = '12pt'
        fig[comp].xaxis.axis_label_text_font_size = '12pt' #label font size adjustment
        fig[comp].yaxis.axis_label_text_font_size = '12pt'
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
        # Special plots
        if numDims == 1:
            x = 0.5*(grid[0][1:]+grid[0][:-1])
            fig[comp].line(x, values[..., comp], line_width=2, legend=label)
        elif numDims == 2:
            if contour:
                pass
            elif streamline:
                magnitude = np.sqrt(values[..., 2*comp]**2 
                                    + values[..., 2*comp+1]**2)
                gridCC = _gridNodalToCellCentered(grid, cells)
                plt.subplots(numRows, numCols,
                             sharex=True, sharey=True)
                strm = plt.streamplot(gridCC[0]*xscale, gridCC[1]*yscale,  # make streamline plot by matplotlib first
                                      values[..., 2*comp].transpose(),
                                      values[..., 2*comp+1].transpose(), 
                                      color=magnitude.transpose(), 
                                      linewidth=2)
                lines = strm.lines # get the line and color data of matplotlib streamline
                pathes = lines.get_paths()
                arr = lines.get_array().data
                points = np.stack([p.vertices.T for p in pathes], axis=0)
                X = points[:, 0, :].tolist()
                Y = points[:, 1, :].tolist()
                mapper = bt.linear_cmap(field_name="color", palette=Inferno256, low=arr.min(), high=arr.max())
                # use the data to create a multiline, use linear_map and palette to set the color of the lines:
                source = bm.ColumnDataSource(dict(x=X, y=Y, color=arr))
                fig[comp].multi_line("x", "y", line_color=mapper, source=source, line_width=3) 
                                    
                #xs, ys = streamlines(gridCC[0]*xscale, 
                #                     gridCC[1]*yscale, # 1d arrays
                #                     values[..., 2*comp].transpose(), # x velocity
                #                     values[..., 2*comp+1].transpose(), # y velocity
                #                     density=1)
                #fig[comp].multi_line(xs, ys, color=inferno(len(xs)), line_width=2, line_alpha=0.8)
                colormapper = bm.LinearColorMapper(palette='Inferno256',
                                    low=np.amin(magnitude.transpose()), 
                                    high=np.amax(magnitude.transpose()))#adding a color bar
                color_bar = bm.ColorBar(color_mapper=colormapper, 
                                 width=7, 
                                 location=(0,0), 
                                 formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                 ticker=bm.BasicTicker(desired_num_ticks=4), 
                                 label_standoff=12, 
                                 border_line_color=None,
                                 padding=2,
                                 bar_line_color='black')
                fig[comp].add_layout(color_bar, 'right')
                #end
            elif quiver:
                gridCC = _gridNodalToCellCentered(grid, cells)
                x_range = gridCC[0]*xscale #setting x coordinates
                y_range = gridCC[1]*yscale #setting y coordinates
                dx = grid[0][1]-grid[0][0]
                dy = grid[1][1]-grid[1][0]
                freq = 2
                v_x = values[..., 2*comp].transpose()
                v_y = values[..., 2*comp+1].transpose()
                X, Y = np.meshgrid(x_range, y_range)
                speed = np.sqrt(v_x**2 + v_y**2)
                theta = np.arctan2(v_y*dy,v_x*dx) #arctan(y/x)
                maxSpeed = speed.max()
                x0 = X[::freq, ::freq].flatten()
                y0 = Y[::freq, ::freq].flatten()
                length = speed[::freq, ::freq].flatten()/ maxSpeed
                angle = theta[::freq, ::freq].flatten()
                x1 = x0 + 0.9 * freq * dx * v_x[::freq, ::freq].flatten()/speed[::freq, ::freq].max()
                y1 = y0 + 0.9 * freq * dy * v_y[::freq, ::freq].flatten()/speed[::freq, ::freq].max()
                fig[comp].segment(x0, y0, x1, y1, color='black') #vector line
                fig[comp].triangle(x1, y1, size=4.0, angle=angle-np.pi/2, color='black') #vector arrow

            elif diverging:
                gridCC = _gridNodalToCellCentered(grid, cells)
                vmax = np.abs(values[..., comp]).max()
                x_range = gridCC[0]*xscale #setting x coordinates
                y_range = gridCC[1]*yscale #setting y coordinates
                CmToRgb = (255 * cm.RdBu_r(range(256))).astype('int') #extract colors from maplotlib colormap
                RgbToHexa = [bc.RGB(*tuple(rgb)).to_hex() for rgb in CmToRgb] # convert RGB numbers into colormap hexacode string
                mapper = bm.LinearColorMapper(palette=RgbToHexa,
                                           low=-vmax, 
                                           high=vmax)#adding a color bar
                fig[comp].image(image=[values[..., comp].transpose()],
                                x=x_range[0], y=y_range[0],
                                dw=(x_range[-1]-x_range[0]), dh=(y_range[-1]-y_range[0]),
                                color_mapper=mapper)    
                color_bar = bm.ColorBar(color_mapper=mapper, 
                                    width=7, 
                                    location=(0,0), 
                                    formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                    ticker=bm.BasicTicker(desired_num_ticks=4), 
                                    label_standoff=12, 
                                    border_line_color=None,
                                    padding=2,
                                    bar_line_color='black')
                fig[comp].add_layout(color_bar, 'right')
            #end
        # Basic  plots
            else:
                gridCC = _gridNodalToCellCentered(grid, cells)
                x_range = gridCC[0]*xscale #setting x coordinates
                y_range = gridCC[1]*yscale #setting y coordinates
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
                    if vmin is not None:
                        vminTemp = vmin
                    else:
                        vminTemp = np.amin(values[...,comp])
                    #end
                    if vmax is not None:
                        vmaxTemp = vmax
                    else:
                        vmaxTemp = np.amax(values[...,comp])

                    mapper = bm.LogColorMapper(palette='Inferno256',
                                            low=MinimumValue, 
                                            high=MaximumValue) 
                    fig[comp].image(image=[tmp.transpose()],
                                    x=x_range[0], y=y_range[0],
                                    dw=(x_range[-1]-x_range[0]), dh=(y_range[-1]-y_range[0]),
                                    color_mapper=mapper)
                    color_bar = bm.ColorBar(color_mapper=mapper, #adding a colorbar
                                        width=7, 
                                        location=(0,0), 
                                        formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                        ticker=bm.BasicTicker(), 
                                        label_standoff=12, 
                                        border_line_color=None,
                                        padding=2,
                                        bar_line_color='black')
                    fig[comp].add_layout(color_bar, 'right')
                else:
                    mapper = bm.LinearColorMapper(palette='Inferno256',
                                            low=np.amin(values[...,comp]), 
                                            high=np.amax(values[...,comp]))                              
                    fig[comp].image(image=[values[..., comp].transpose()],
                                    x=x_range[0], y=y_range[0],
                                    dw=(x_range[-1]-x_range[0]), dh=(y_range[-1]-y_range[0]),
                                    color_mapper=mapper)
                    color_bar = bm.ColorBar(color_mapper=mapper, #adding a colorbar
                                        width=7, 
                                        location=(0,0), 
                                        formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                        ticker=bm.BasicTicker(desired_num_ticks=4), 
                                        label_standoff=12, 
                                        border_line_color=None,
                                        padding=2,
                                        bar_line_color='black',
                                        major_label_text_font_size='9pt')
                    fig[comp].add_layout(color_bar, 'right')
                #end
            #end
        else:
            raise ValueError("{:d}D data not yet supported".
                             format(numDims))
        #end

        #-------------------------------------------------------------
        #-- Additional Formatting ------------------------------------
    
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
        if legend is False:
            fig[comp].legend.visible = False
        #end
    #end
        if logx:
            pass
        #end
        if logy:
            pass
        #end
    #for i in range(numComps, len(ax)):
    #    ax[i].axis('off')
    ##end

    gp = bl.gridplot(children=fig, toolbar_location='right', ncols=numCols, merge_tools=True)
    #blt.output_file("image.html", title="Postgkyl output")
    #blt.show(gp)
    return gp
#end
