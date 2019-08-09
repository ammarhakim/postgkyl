import matplotlib.cm as cm
import matplotlib.pyplot as plt
import click
import numpy as np
import os.path
import bokeh.plotting as blt
import bokeh.models as bm
import bokeh.layouts as bl
import bokeh.transform as bt
import bokeh.colors as bc
import bokeh.palettes as bp


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
                tooltips = None
            else:
                tooltips = [(axLabel[0], "$x"), (axLabel[1], "$y"), ("value", "@image")]
            #end
        #end                               
        for comp in idxComps:
            if logx and logy:
                fig.append(blt.figure(tooltips=tooltips,
                                      x_axis_type="log",
                                      y_axis_type="log",
                                      frame_height=int(600.0/numRows),
                                      frame_width=int(600.0/numRows),
                                      outline_line_color='black',
                                      min_border_left=70,
                                      min_border_right=50,
                                      min_border_bottom=10))
            elif logx:
                fig.append(blt.figure(tooltips=tooltips,
                                      x_axis_type = "log",
                                      frame_height=int(600.0/numRows),#adjust figures with the size based on the screen size
                                      frame_width=int(600.0/numRows),
                                      outline_line_color='black',
                                      min_border_left=70,
                                      min_border_right=50,
                                      min_border_bottom=10)) #adjust spacings betweewn subplots to be aligned
            elif logy:
                fig.append(blt.figure(tooltips=tooltips,
                                         y_axis_type = "log",
                                          frame_height=int(600.0/numRows),
                                          frame_width=int(600.0/numRows),
                                          outline_line_color='black',
                                          min_border_left=70,
                                          min_border_right=50,
                                          min_border_bottom=10))
            else:
                fig.append(blt.figure(tooltips=tooltips,
                                      frame_height=int(600.0/numRows),
                                      frame_width=int(600.0/numRows),
                                      outline_line_color='black',
                                      min_border_left=70,
                                      min_border_right=60,
                                      min_border_bottom=10))
                        
    #-- Preparing the Axes -------------------------------------------
    for comp in idxComps:
        fig[comp].xaxis.minor_tick_line_color = None #deleting minor ticks
        fig[comp].yaxis.minor_tick_line_color = None

        fig[comp].xaxis.major_label_text_font_size = '12pt' #tick font size adjustment
        fig[comp].yaxis.major_label_text_font_size = '12pt'

        fig[comp].xaxis.axis_label_text_font_size = '12pt' #label font size adjustment
        fig[comp].yaxis.axis_label_text_font_size = '12pt'

        fig[comp].xaxis.formatter = bm.BasicTickFormatter(precision=1)#adjust floating numbers of ticks
        fig[comp].yaxis.formatter = bm.BasicTickFormatter(precision=1)

        if numDims != 1:
            if comp % numCols != 0: #hiding labels for unnecessary locations
                fig[comp].yaxis.major_label_text_font_size = '0pt'
            #end
            if comp < (numRows-1) * numCols:
                fig[comp].xaxis.major_label_text_font_size = '0pt'
            #end
        #end
        if comp >= (numRows-1) * numCols:
            if xlabel is None:
                fig[comp].xaxis.axis_label = axLabel[0]
            else: #if there is xlabel to be specified
                fig[comp].xaxis.axis_label = xlabel
            #end
            #end
        #end

        if comp % numCols == 0:
            if ylabel is None:
                if numDims == 2:
                    fig[comp].yaxis.axis_label = axLabel[1]
                #end
            else:
                fig[comp].yaxis.axis_label = ylabel
            #end
        #end
    #end

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
                mapper = bt.linear_cmap(field_name="color", palette=bp.Inferno256, low=arr.min(), high=arr.max())
                # use the data to create a multiline, use linear_map and palette to set the color of the lines:
                source = bm.ColumnDataSource(dict(x=X, y=Y, color=arr))
                fig[comp].multi_line("x", "y", line_color=mapper, source=source, line_width=3) 
                colormapper = bm.LinearColorMapper(palette='Inferno256',
                                    low=np.amin(magnitude.transpose()), 
                                    high=np.amax(magnitude.transpose()))#adding a color bar
                color_bar = bm.ColorBar(color_mapper=colormapper, 
                                 width=7, 
                                 location=(0,0), 
                                 formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                 ticker=bm.BasicTicker(desired_num_ticks=4), 
                                 label_standoff=13,
                                 major_label_text_font_size='12pt', 
                                 border_line_color=None,
                                 padding=2,
                                 bar_line_color='black')
                fig[comp].add_layout(color_bar, 'right')
            elif quiver:
                gridCC = _gridNodalToCellCentered(grid, cells)
                x_range = gridCC[0]*xscale #setting x coordinates
                y_range = gridCC[1]*yscale #setting y coordinates
                dx = grid[0][1]-grid[0][0]
                dy = grid[1][1]-grid[1][0]
                freq = 7
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
                x_range = grid[0]*xscale #setting x coordinates
                y_range = grid[1]*yscale #setting y coordinates
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
                                    label_standoff=14,
                                    major_label_text_font_size='12pt',
                                    border_line_color=None,
                                    padding=2,
                                    bar_line_color='black')
                fig[comp].add_layout(color_bar, 'right')
        # Basic  plots
            else:
                gridCC = _gridNodalToCellCentered(grid, cells)
                x_range = grid[0]*xscale #setting x coordinates
                y_range = grid[1]*yscale #setting y coordinates
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
                    #end
                    mapper = bm.LogColorMapper(palette='Inferno256',
                                            low=vminTemp, 
                                            high=vmaxTemp) 
                    fig[comp].image(image=[tmp.transpose()],
                                    x=x_range[0], y=y_range[0],
                                    dw=(x_range[-1]-x_range[0]), dh=(y_range[-1]-y_range[0]),
                                    color_mapper=mapper)
                    color_bar = bm.ColorBar(color_mapper=mapper, #adding a colorbar
                                        width=7, 
                                        location=(0,0), 
                                        formatter=bm.BasicTickFormatter(precision=1), #deleting unnecessary floating numbers
                                        ticker=bm.BasicTicker(), 
                                        label_standoff=13,
                                        major_label_text_font_size='12pt', 
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
                                        label_standoff=13,
                                        major_label_text_font_size='12pt', 
                                        border_line_color=None,
                                        padding=2,
                                        bar_line_color='black')
                    fig[comp].add_layout(color_bar, 'right')
                #end
            #end
        else:
            raise ValueError("{:d}D data not yet supported".
                             format(numDims))
        #end

        #-- Additional Formatting ------------------------------------
        if not quiver:
            fig[comp].x_range.range_padding  = 0
            if numDims == 2:
                fig[comp].y_range.range_padding = 0
            #end
        #end
        
        if legend:
            if numDims == 2:
                x_range = grid[0]*xscale
                y_range = grid[1]*yscale
                #The legends are not embedded into the plot but the text numbers on the top of plots. Refer to 1D line plot.
                legend_number = bm.Label(x=x_range[0]+(x_range[-1]-x_range[0])*0.005, 
                                    y=y_range[-1]-(y_range[-1]-y_range[0])*0.115, 
                                    text=label, render_mode='css',
                                    background_fill_color='white', 
                                    background_fill_alpha=0.9, 
                                    border_line_cap='round')
                        
                fig[comp].add_layout(legend_number)
            #end
        #end
        if title:
            if comp < numCols:
                fig[comp].title.text = title
            #end
        #end

        if not legend:
            if numDims == 1:
                fig[comp].legend.visible = False
            #end
        #end

    gp = bl.gridplot(children=fig, toolbar_location='right', ncols=numCols, merge_tools=True)

    return gp
#end
