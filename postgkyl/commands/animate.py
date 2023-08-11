import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import click

import postgkyl.output.plot as gplot
import postgkyl.data.select as select
from postgkyl.commands.util import verb_print

def _update(i, data, fig, offsets, kwargs):
  fig.clear()
  kwargs['figure'] = fig

  for n in offsets:
    dat = data[i+n]
    kwargs['title'] = ''
    if not kwargs['notitle']:
      if dat.ctx['frame'] is not None:
        kwargs['title'] = kwargs['title'] + 'F: {:d} '.format(dat.ctx['frame'])
      #end
      if dat.ctx['time'] is not None:
        kwargs['title'] = kwargs['title'] + 'T: {:.4e}'.format(dat.ctx['time'])
      #end
    #end
    if kwargs['arg'] is not None:
      im = gplot(dat, kwargs['arg'], **kwargs)
    else:
      im = gplot(dat, **kwargs)
    #end
  #end
  return(im)
#end

@click.command()
@click.option('--use', '-u', default=None,
              help="Specify a tag to plot.")
@click.option('--grouptags', is_flag=True,
              help="Group coresponding tagged frames.")
@click.option('--squeeze', '-s', is_flag=True,
              help="Squeeze the components into one panel.")
@click.option('--subplots', '-b', is_flag=True,
              help="Make subplots from multiple datasets.")
@click.option('--nsubplotrow', 'nSubplotRow', type=click.INT,
              help="Manually set the number of rows for subplots.")
@click.option('--nsubplotcol', 'nSubplotCol', type=click.INT,
              help="Manually set the number of columns for subplots.")
@click.option('--transpose', is_flag=True,
              help="Transpose axes.")
@click.option('-c', '--contour', is_flag=True,
              help="Make contour plot.")
@click.option('--clevels', type=click.STRING,
              help="Specify levels for contours: either integer or start:end:nlevels")
@click.option('-q', '--quiver', is_flag=True,
              help="Make quiver plot.")
@click.option('-l', '--streamline', is_flag=True,
              help="Make streamline plot.")
@click.option('--sdensity', type=click.FLOAT,
              help="Control density of the streamlines.")
@click.option('--arrowstyle', type=click.STRING,
              help="Set the style for streamline arrows.")
@click.option('-g', '--group', type=click.Choice(['0', '1']),
              help="Switch to group mode.")
@click.option('-s', '--scatter', is_flag=True,
              help="Make scatter plot.")
@click.option('--markersize', type=click.FLOAT,
              help="Set marker size for scatter plots.")
@click.option('--linewidth', type=click.FLOAT,
              help="Set the linewidth.")
@click.option('--color', type=click.STRING,
              help="Set color when available.")
@click.option('--style',
              help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('-d', '--diverging', is_flag=True,
              help="Switch to diverging colormesh mode.")
@click.option('--arg', type=click.STRING,
              help="Additional plotting arguments, e.g., '*--'.")
@click.option('-a', '--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
@click.option('--logx', is_flag=True,
              help="Set x-axis to log scale.")
@click.option('--logy', is_flag=True,
              help="Set y-axis to log scale.")
@click.option('--logz', is_flag=True,
              help="Set values of 2D plot to log scale.")
@click.option('--xshift', default=0.0, type=click.FLOAT,
              help="Value to shift the x-axis (default: 0.0).")
@click.option('--yshift', default=0.0, type=click.FLOAT,
              help="Value to shift the y-axis (default: 0.0).")
@click.option('--zshift', default=0.0, type=click.FLOAT,
              help="Value to shift the z-axis (default: 0.0).")
@click.option('--xscale', default=1.0, type=click.FLOAT,
              help="Value to scale the x-axis (default: 1.0).")
@click.option('--yscale', default=1.0, type=click.FLOAT,
              help="Value to scale the y-axis (default: 1.0).")
@click.option('--zscale', default=1.0, type=click.FLOAT,
              help="Value to scale the z-axis (default: 1.0).")
@click.option('--float', is_flag=True,
              help="Choose min/max levels based on current frame (i.e., each frame uses a different color range).")
@click.option('--xmax', default=None, type=click.FLOAT,
              help="Set maximal x-value.")
@click.option('--xmin', default=None, type=click.FLOAT,
              help="Set minimal x-values.")
@click.option('--ymax', default=None, type=click.FLOAT,
              help="Set maximal y-value.")
@click.option('--ymin', default=None, type=click.FLOAT,
              help="Set minimal y-values.")
@click.option('--zmax', default=None, type=click.FLOAT,
              help="Set maximal z-value.")
@click.option('--zmin', default=None, type=click.FLOAT,
              help="Set minimal z-values.")
@click.option('--xlim', default=None, type=click.STRING,
              help="Set limits for the x-coordinate (lower,upper)")
@click.option('--ylim', default=None, type=click.STRING,
              help="Set limits for the y-coordinate (lower,upper).")
@click.option('--zlim', default=None, type=click.STRING,
              help="Set limits for the z-coordinate (lower,upper).")
@click.option('--legend/--no-legend', default=True,
              help="Show legend.")
@click.option('--force-legend', 'forcelegend', is_flag=True,
              help="Force legend even when plotting a single dataset.")
@click.option('-x', '--xlabel', type=click.STRING,
              help="Specify a x-axis label.")
@click.option('-y', '--ylabel', type=click.STRING,
              help="Specify a y-axis label.")
@click.option('--clabel', type=click.STRING,
              help="Specify a label for colorbar.")
@click.option('--title', type=click.STRING,
              help="Specify a title.")
@click.option('--notitle', is_flag=True,
              help="Do not show title.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
@click.option('--fps', type=click.INT,
              help="Specify frames per second for saving.")
@click.option('--dpi', type=click.INT,
              help="DPI (resolution) for output.")
@click.option('-e', '--edgecolors', type=click.STRING,
              help="Set color for cell edges (default: None)")
@click.option('--showgrid/--no-showgrid', default=True,
              help="Show grid-lines (default: True)")
@click.option('--collected', is_flag=True,
              help="Animate a dataset that has been collected, i.e. a single dataset with time taken to be the first index.")
@click.option('--hashtag', is_flag=True,
              help="Turns on the pgkyl hashtag!")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
@click.option('--saveframes', type=click.STRING,
              help="Save individual frames as PNGS instead of an animation")
@click.option('--figsize',
              help="Comma-separated values for x and y size.")
@click.pass_context
def animate(ctx, **kwargs):
  r"""Animate the actively loaded dataset and show resulting plots in a
  loop. Typically, the datasets are loaded using wildcard/regex
  feature of the -f option to the main pgkyl executable. To save the
  animation ffmpeg needs to be installed.

  """
  verb_print(ctx, 'Starting animate')
  data = ctx.obj['data']

  if kwargs['xlim']:
    kwargs['xmin'] = float(kwargs['xlim'].split(',')[0])
    kwargs['xmax'] = float(kwargs['xlim'].split(',')[1])
  #end
  if kwargs['ylim']:
    kwargs['ymin'] = float(kwargs['ylim'].split(',')[0])
    kwargs['ymax'] = float(kwargs['ylim'].split(',')[1])
  #end
  if kwargs['zlim']:
    kwargs['zmin'] = float(kwargs['zlim'].split(',')[0])
    kwargs['zmax'] = float(kwargs['zlim'].split(',')[1])
  #end

  if not kwargs['float']:
    vmin = float('inf')
    vmax = float('-inf')
    for dat in ctx.obj['data'].iterator(kwargs['use']):
      num_dims = dat.getNumDims()
      if num_dims == 1:
        val = dat.getValues()*kwargs['yscale']
      elif num_dims == 2:
        val = dat.getValues()*kwargs['zscale']
      #end
      if vmin > np.nanmin(val):
        vmin = np.nanmin(val)
      if vmax < np.nanmax(val):
        vmax = np.nanmax(val)
      #end
    #end

    if num_dims == 1:
      if kwargs['ymin'] is None:
        kwargs['ymin'] = vmin
      #end
      if kwargs['ymax'] is None:
        kwargs['ymax'] = vmax
      #end
    elif num_dims == 2:
      if kwargs['zmin'] is None:
        kwargs['zmin'] = vmin
      #end
      if kwargs['zmax'] is None:
        kwargs['zmax'] = vmax
      #end
    #end
  #end

  anims = []
  figs = []
  kwargs['legend'] = False

  figsize = None
  if kwargs['figsize']:
    figsize = (int(kwargs['figsize'].split(',')[0]),
               int(kwargs['figsize'].split(',')[1]))
  #end

  offsets = [0]
  tagIterator = list(data.tagIterator(kwargs['use']))
  setFigure = False
  minSize = np.NAN

  if kwargs['grouptags']:
    for tag in data.tagIterator(kwargs['use']):
      numDatasets = int(data.getNumDatasets(tag=tag))
      offsets.append(numDatasets)
      minSize = int(np.nanmin((minSize, numDatasets)))
    #end
    offsets.pop()
    tagIterator = list((kwargs['use'],))
    kwargs['legend'] = True
    setFigure = True
    figNum = int(0)

    for tag in tagIterator:
      dataList = list(data.iterator(tag=tag))
      if setFigure:
        figs.append(plt.figure(figNum, figsize=figsize))
      else:
        figs.append(plt.figure(figsize=figsize))
      #end
      if not kwargs['saveframes']:
        anims.append(FuncAnimation(figs[-1], _update,
                                   int(np.nanmin((minSize, len(dataList)))),
                                   fargs=(dataList, figs[-1],
                                          offsets, kwargs),
                                   interval=kwargs['interval'], blit=False))

        if tag is not None:
          fName = 'anim_{:s}.mp4'.format(tag)
        else:
          fName = 'anim.mp4'
        #end
        if kwargs['saveas']:
          fName = str(kwargs['saveas'])
        #end
        if kwargs['save'] or kwargs['saveas']:
          anims[-1].save(fName, writer='ffmpeg',
                         fps=kwargs['fps'], dpi=kwargs['dpi'])
        #end
      else:
        for i in range(int(np.nanmin((minSize, len(dataList))))):
          _update(i, dataList, figs[-1], offsets, kwargs)
          plt.savefig('{:s}_{:d}.png'.format(kwargs['saveframes'], i),
                      dpi=kwargs['dpi'])
        #end
        kwargs['show'] = False # do not show in this case
      #end
    #end
  else:
    dataList = list(data.iterator(tag=kwargs['use']))
    if setFigure:
      figs.append(plt.figure(figNum, figsize=figsize))
    else:
      figs.append(plt.figure(figsize=figsize))
    #end
    if not kwargs['saveframes']:
      anims.append(FuncAnimation(figs[-1], _update,
                                 int(np.nanmin((minSize, len(dataList)))),
                                 fargs=(dataList, figs[-1],
                                        offsets, kwargs),
                                 interval=kwargs['interval'], blit=False))

      fName = 'anim.mp4'
      if kwargs['saveas']:
        fName = str(kwargs['saveas'])
      #end
      if kwargs['save'] or kwargs['saveas']:
        anims[-1].save(fName, writer='ffmpeg',
                       fps=kwargs['fps'], dpi=kwargs['dpi'])
      #end
    else:
      for i in range(int(np.nanmin((minSize, len(dataList))))):
        _update(i, dataList, figs[-1], offsets, kwargs)
        plt.savefig('{:s}_{:d}.png'.format(kwargs['saveframes'], i),
                    dpi=kwargs['dpi'])
      #end
      kwargs['show'] = False # do not show in this case
    #end
  #end

  if kwargs['show']:
    plt.show()
  #end
  verb_print(ctx, 'Finishing animate')
#end
