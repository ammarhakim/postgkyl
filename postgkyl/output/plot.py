import os.path

import matplotlib as mpl
from matplotlib import cm
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# this is needed for Python 3.0 compatibility
import sys
if sys.version_info[0] >= 3:
    unicode = str

# Helper functions
def _colorbar(obj, fig, cax, label="", extend=None):
  divider = make_axes_locatable(cax)
  cax2 = divider.append_axes("right", size="3%", pad=0.05)
  #if extend:
  return fig.colorbar(obj, cax=cax2, label=label or "", extend=extend)
  #end
  #return fig.colorbar(obj, cax=cax2, label=label or "")
#end

def _get_nodal_grid(grid, cells):
  num_dims = len(grid)
  grid_out = []
  if num_dims != len(cells):  # sanity check
    raise ValueError("Number dimensions for 'grid' and 'values' doesn't match")
  #end
  for d in range(num_dims):
    if len(grid[d].shape) == 1:
      if grid[d].shape[0] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[0] == cells[d]+1:
        grid_out.append(0.5*(grid[d][:-1] + grid[d][1:]))
      else:
        raise ValueError("Something is terribly wrong...")
      #end
    else:
      if grid[d].shape[d] == cells[d]:
        grid_out.append(grid[d])
      elif grid[d].shape[d] == cells[d]+1:
        if num_dims ==1 :
          grid_out.append(0.5*(grid[d][:-1] + grid[d][1:]))
        else:
          grid_out.append(0.5*(grid[d][:-1,:-1] + grid[d][1:,1:]))
        #end
      else:
        raise ValueError("Something is terribly wrong...")
      #end
    #end
  #end
  return grid_out
#end



def plot(data, args=(),
         figure=None, squeeze=False,
         num_axes=None, start_axes=0,
         num_subplot_row=None, num_subplot_col=None,
         streamline=False, sdensity=1, arrowstyle='simple',
         quiver=False,
         contour=False, clevels=None, cnlevels=None, cont_label=False,
         diverging=False,
         lineouts=None, group=None,
         xmin=None, xmax=None, xscale=1.0, xshift=0.0,
         ymin=None, ymax=None, yscale=1.0, yshift=0.0,
         zmin=None, zmax=None, zscale=1.0, zshift=0.0,
         style=None, rcParams=None,
         legend=True, label_prefix='', colorbar=True,
         xlabel=None, ylabel=None, clabel=None, title=None,
         logx=False, logy=False, logz=False,
         fixaspect=False, aspect=None,
         edgecolors=None, showgrid=True,
         hashtag=False, xkcd=False,
         color=None, markersize=None, linewidth=None,
         #transpose=False, # trasnspose should be done before plotting
         figsize=None,
         jet=False,
         cmap=None,
         **kwargs):
  """Plots Gkeyll data

  Unifies the plotting across a wide range of Gkyl applications. Can
  be used for both 1D an 2D data. Uses a proper colormap by default.
  """

  if group is not None:
    lineouts=group
    print("plot.py Deprecation warning: the 'group' parameter is being renamed to 'lineouts', which is hopefully more explanatory.")
  #end

  #---- Set style and process inputs -----------------------------------
  # Default to Postgkyl style file file if no style is specified
  # Use the rcParams dictionary which is passed with click contex
  if bool(style):
    plt.style.use(style)
  elif bool(rcParams):
    for key in rcParams:
      mpl.rcParams[key] = rcParams[key]
    #end
  else:
    plt.style.use(os.path.dirname(os.path.realpath(__file__)) +
                  "/postgkyl.mplstyle")
  #end

  # Process input parameters
  if not bool(aspect):
    aspect = 1.0
  #end

  if bool(cmap):
    mpl.rcParams['image.cmap'] = cmap
  elif bool(diverging):
    mpl.rcParams['image.cmap'] = 'RdBu_r'
  #end

  # This should not be used on its own; however, it can be useful for
  # comparing results with literature
  if bool(jet):
    mpl.rcParams['image.cmap'] = 'jet'
  #end

  # The most important thing
  if xkcd:
    plt.xkcd()
  #end

  if not bool(color):
    cl = data.color
  #end
  if bool(color):
    mpl.rcParams['lines.color'] = color
  #end
  if bool(linewidth):
    mpl.rcParams['lines.linewidth'] = linewidth
  #end

  #---- Data Loading ---------------------------------------------------
  num_dims = data.getNumDims(squeeze=True)
  if num_dims > 2:
    raise Exception('Only 1D and 2D plots are currently supported')
  #end
  # Get the handles on the grid and values
  grid = data.getGrid().copy()
  values = data.getValues()
  lower, upper = data.getBounds()
  cells = data.getNumCells()

  # Squeeze the data (get rid of "collapsed" dimensions)
  axes_labels = ['$z_0$', '$z_1$', '$z_2$', '$z_3$', '$z_4$', '$z_5$']
  if len(grid) > num_dims:
    idx = []
    for dim, g in enumerate(grid):
      if cells[dim] <= 1:
        idx.append(dim)
      #end
      grid[dim] = g.squeeze()
    #end
    if bool(idx):
      for i in reversed(idx):
        grid.pop(i)
      #end
      lower = np.delete(lower, idx)
      upper = np.delete(upper, idx)
      cells = np.delete(cells, idx)
      axes_labels = np.delete(axes_labels, idx)
      values = np.squeeze(values, tuple(idx))
    #end
  #end

  # Get the number of components and an indexer
  step = 2 if bool(streamline or quiver) else 1
  num_comps = values.shape[-1]
  idx_comps = range(int(np.floor(num_comps/step)))
  if num_axes:
    num_comps = num_axes
  else:
    num_comps = len(idx_comps)
  #end

  # Create axis labels
  if xlabel is None:
    xlabel = axes_labels[0] if lineouts != 1 else axes_labels[1]
    if xshift != 0.0 and xscale != 1.0:
      xlabel = r'({:s} + {:.2e}) $\times$ {:.2e}'.format(xlabel, xshift, xscale)
    elif xshift != 0.0:
      xlabel = r'{:s} + {:.2e}'.format(xlabel, xshift)
    elif xscale != 1.0:
      xlabel = r'{:s} $\times$ {:.2e}'.format(xlabel, xscale)
    #end
  #end
  if ylabel is None and num_dims == 2 and lineouts is None:
    ylabel = axes_labels[1]
    if yshift != 0.0 and yscale != 1.0:
      ylabel = r'({:s} + {:.2e}) $\times$ {:.2e}'.format(ylabel, yshift, yscale)
    elif xshift != 0.0:
      ylabel = r'{:s} + {:.2e}'.format(ylabel, yshift)
    elif xscale != 1.0:
      ylabel = r'{:s} $\times$ {:.2e}'.format(ylabel, yscale)
    #end
  #end
  if zscale != 1.0:
    if clabel:
      clabel = clabel + r' $\times$ {:.3e}'.format(zscale)
    else:
      clabel = r'$\times$ {:.3e}'.format(zscale)
    #end
  #end


  #---- Prepare Figure and Axes ----------------------------------------
  if bool(figsize):
    figsize = (int(figsize.split(',')[0]),
               int(figsize.split(',')[1]))
  #end
  if figure is None:
    fig = plt.figure(figsize=figsize)
  elif isinstance(figure, int):
    fig = plt.figure(figure, figsize=figsize)
  elif isinstance(figure, matplotlib.figure.Figure):
    fig = figure
  elif isinstance(figure, (str, unicode)):
    fig = plt.figure(int(figure), figsize=figsize)
  else:
    raise TypeError('\'fig\' keyword needs to be one of ' \
                    'None (default), int, or MPL Figure')
  #end

  # Axes
  if fig.axes:
    ax = fig.axes
    if squeeze is False and num_comps > len(ax):
      raise ValueError(
        "Trying to plot into figure with not enough axes")
    #end
  else:
    if squeeze:  # Plotting into 1 panel
      plt.subplots(1, 1, num=fig.number)
      ax = fig.axes
      ax[0].set_xlabel(xlabel)
      ax[0].set_ylabel(ylabel)
      if title is not None:
        ax[0].set_title(title, y=1.08)
      #end
    else:  # Plotting each components into its own subplot
      if num_subplot_row is not None:
        num_rows = num_subplot_row
        num_cols = int(np.ceil(num_comps/num_rows))
      elif num_subplot_col is not None:
        num_cols = num_subplot_col
        num_rows = int(np.ceil(num_comps/num_cols))
      else:
        sr = np.sqrt(num_comps)
        if sr == np.ceil(sr):
          num_rows = int(sr)
          num_cols = int(sr)
        elif np.ceil(sr) * np.floor(sr) >= num_comps:
          num_rows = int(np.floor(sr))
          num_cols = int(np.ceil(sr))
        else:
          num_rows = int(np.ceil(sr))
          num_cols = int(np.ceil(sr))
        #end
      #end

      if num_dims == 1 or lineouts is not None:
        plt.subplots(num_rows, num_cols,
                     sharex=True,
                     num=fig.number)
      else: # In 2D, share y-axis as well
        plt.subplots(num_rows, num_cols,
                     sharex=True, sharey=True,
                     num=fig.number)
      #end
      ax = fig.axes
      # Removing extra axes
      for i in range(num_comps, len(ax)):
        ax[i].axis('off')
      #end
      # Adding labels only to the right subplots
      for ax_idx, _ in enumerate(ax):
        if ax_idx >= (num_rows-1) * num_cols:
          ax[ax_idx].set_xlabel(xlabel)
        #end
        if ax_idx % num_cols == 0:
          ax[ax_idx].set_ylabel(ylabel)
        #end
        if ax_idx < num_cols and bool(title):
          ax[ax_idx].set_title(title, y=1.08)
        #end
      #end
    #end
  #end

  #---- Main Plotting Loop ---------------------------------------------
  for comp in idx_comps:
    cax = ax[0] if squeeze else ax[comp+start_axes]
    label = ('{:s}_c{:d}'.format(label_prefix, comp)).strip('_') if len(idx_comps) > 1 else label_prefix

    if num_dims == 1:
      nodal_grid = _get_nodal_grid(grid, cells)
      x = (nodal_grid[0] + xshift) * xscale
      y = (values[..., comp] + yshift) * yscale
      im = cax.plot(x, y,
                    *args, color=color, label=label, markersize=markersize)


    elif num_dims == 2:
      extend = None

      if contour:  #----------------------------------------------------
        levels = 10
        if cnlevels:
          levels = int(cnlevels)-1
        elif clevels:
          if ":" in clevels:
            s = clevels.split(":")
            levels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
          else:
            levels = np.array(clevels.split(','))
          #end
        #end
        nodal_grid = _get_nodal_grid(grid, cells)
        x = (nodal_grid[0] + xshift) * xscale
        y = (nodal_grid[1] + yshift) * yscale
        z = (values[..., comp].transpose() + zshift) * zscale
        im = cax.contour(x, y, z,
                         levels, *args,
                         colors=color, linewidths=linewidth)
        if cont_label:
          cax.clabel(im, inline=1)
        #end


      elif quiver: #----------------------------------------------------
        skip = int(np.max((len(grid[0]), len(grid[1])))//15)
        skip2 = int(skip//2)
        nodal_grid = _get_nodal_grid(grid, cells)
        x = (nodal_grid[0][skip2::skip] + xshift) * xscale
        y = (nodal_grid[1][skip2::skip] + yshift) * yscale
        z1 = (values[skip2::skip, skip2::skip, 2*comp].transpose()
              + zshift) * zscale
        z2 = (values[skip2::skip, skip2::skip, 2*comp+1].transpose()
              + zshift) * zscale
        im = cax.quiver(x, y, z1, z2)


      elif streamline: #------------------------------------------------
        if bool(color):
          cl = color
        else:
          # magnitude
          cl = np.sqrt(values[..., 2*comp]**2
                       + values[..., 2*comp+1]**2).transpose()
        #end
        nodal_grid = _get_nodal_grid(grid, cells)
        x = (nodal_grid[0] + xshift) * xscale
        y = (nodal_grid[1] + yshift) * yscale
        z1 = (values[..., 2*comp].transpose() + zshift) * zscale
        z2 = (values[..., 2*comp+1].transpose() + zshift) * zscale
        print(x.shape, y.shape, z1.shape, z2.shape)
        im = cax.streamplot(x, y, z1, z2,
                            *args,
                            density=sdensity,
                            color=cl, linewidth=linewidth)


      elif lineouts is not None:  #-------------------------------------
        num_lines = values.shape[1] if lineouts == 0 else values.shape[0]
        nodal_grid = _get_nodal_grid(grid, cells)

        if lineouts == 0:
          x = (nodal_grid[0] + xshift) * xscale
          vmin = (nodal_grid[1][0] + yshift) * yscale
          vmax = (nodal_grid[1][-1] + yshift) * yscale
          label = clabel or axes_labels[1]
        else:
          x = (nodal_grid[1] + xshift) * xscale
          vmin = (nodal_grid[0][0] + yshift) * yscale
          vmax = (nodal_grid[0][-1] + yshift) * yscale
          label = clabel or axes_labels[0]
        #end
        idx = [slice(0, u) for u in values.shape]
        idx[-1] = comp
        for line in range(num_lines):
          color = cm.inferno(line / (num_lines-1))
          if lineouts == 0:
            idx[1] = line
          else:
            idx[0] = line
          #end
          y = (values[tuple(idx)] + yshift) * yscale
          im = cax.plot(x, y, *args, color=color)
        #end
        mappable = cm.ScalarMappable(
          norm=colors.Normalize(vmin=vmin, vmax=vmax,
                                clip=False),
          cmap=cm.inferno)
        cb = _colorbar(mappable, fig, cax, label=label)
        colorbar = False
        legend = False


      else: #-----------------------------------------------------------
        if zmin is not None and zmax is not None:
          extend = 'both'
        elif zmax is not None:
          extend = 'max'
        elif zmin is not None:
          extend = 'min'
        #end
        x = (grid[0] + xshift) * xscale
        y = (grid[1] + yshift) * yscale
        z = (values[..., comp].transpose() + zshift) * zscale
        if len(x) == z.shape[1] or len(y) == z.shape[0]:
          nodal_grid = _get_nodal_grid(grid, cells)
          x = (nodal_grid[0] + xshift) * xscale
          y = (nodal_grid[1] + yshift) * yscale
        #end
        if len(x.shape) > 1:
          x, y = x.transpose(), y.transpose()
        #end
        if diverging:
          zmax = np.abs(z).max()
          zmin = -zmax
        #end
        vmax, vmin = zmax, zmin
        norm = None
        if logz:
          if diverging:
            tmp = vmax/1000
            norm = colors.SymLogNorm(linthresh=tmp, linscale=tmp,
                                     vmin=vmin, vmax=vmax, base=10)
          else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
          #end
          vmin, vmax = None, None
        #end
        im = cax.pcolormesh(x, y, z,
                            norm=norm, vmin=vmin, vmax=vmax,
                            edgecolors=edgecolors,
                            linewidth=0.1, shading='auto',
                            *args)
      #end
      if not bool(color) and colorbar and streamline is None:
        cb = _colorbar(im, fig, cax, extend=extend, label=clabel)
      #end
    else:
      raise ValueError("{:d}D data not supported".
                       format(num_dims))
    #end


    #---- Additional Formatting ----------------------------------------
    cax.grid(showgrid)
    # Legend
    if legend:
      if num_dims == 1 and label != '':
        cax.legend(loc=0)
      else:
        cax.text(0.03, 0.96, label,
                 bbox={'facecolor':'w', 'edgecolor':'w', 'alpha':0.8,
                       'boxstyle':'round'},
                 verticalalignment='top',
                 horizontalalignment='left',
                 transform=cax.transAxes)
      #end
    #end
    if hashtag:
      cax.text(0.97, 0.03, '#pgkyl',
               bbox={'facecolor':'w', 'edgecolor':'w', 'alpha':0.8,
                     'boxstyle':'round'},
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
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y')
    if xmin is not None or xmax is not None:
      cax.set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
      cax.set_ylim(ymin, ymax)

    if num_dims == 2:
      if fixaspect:
        plt.setp(cax, aspect=aspect)
      #end
    #end
  #end

  plt.tight_layout()
  return im
#end
