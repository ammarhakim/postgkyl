import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import click

import postgkyl.output.plot as gplot
from postgkyl.commands.util import verb_print

def update(i, ax, ctx, leap, vel,
           xmin, xmax, ymin, ymax, zmin, zmax,
           tag):
  colors = ['C0', 'C1', 'C2', 'C3', 'C4',
            'C5', 'C6', 'C7', 'C8', 'C9']

  s = -1
  plt.cla()
  #for s, dat in ctx.obj['data'].iterator(tag, emum=True):
  for dat in ctx.obj['data'].iterator(tag):
    s = s+1
    time = dat.get_grid()[0]
    coords = dat.get_values()
    tIdx = int(i*leap)

    if xmin is not None:
      x = np.where(coords[:, 0]>xmin, coords[:, 0], np.nan)
    else:
      x = coords[:,0]
    #end
    if xmax is not None:
      x = np.where(x<xmax, x, np.nan)
    #end
    if ymin is not None:
      y = np.where(coords[:, 1]>ymin, coords[:, 1], np.nan)
    else:
      y = coords[:,1]
    #end
    if ymax is not None:
      y = np.where(y<ymax, y, np.nan)
    #end
    if zmin is not None:
      z = np.where(coords[:, 2]>zmin, coords[:, 2], np.nan)
    else:
      z = coords[:,2]
    #end
    if zmax is not None:
      z = np.where(z<zmax, z, np.nan)
    #end

    ax.plot(x, y, z, color=colors[s%10])
    ax.scatter(x[tIdx], y[tIdx], z[tIdx],
               color=colors[s%10])
    if vel and dat.get_num_comps() == 6:
      if tIdx+leap >= len(time) :
        dt = time[-1] - time[tIdx]
      else:
        dt = time[int(tIdx+leap)] - time[tIdx]
      #end
      dx = coords[i, 3]*dt
      dy = coords[i, 4]*dt
      dz = coords[i, 5]*dt
      ax.plot([x[tIdx], x[tIdx]+dx],
              [y[tIdx], y[tIdx]+dy],
              [z[tIdx], z[tIdx]+dz],
              color=colors[s%10])
    #end
  #end
  plt.title('T: {:.4e}'.format(time[tIdx]))
  ax.set_xlabel('$z_0$')
  ax.set_ylabel('$z_1$')
  ax.set_zlabel('$z_2$')
  ax.set_xlim3d(xmin, xmax)
  ax.set_ylim3d(ymin, ymax)
  ax.set_zlim3d(zmin, zmax)
#end

@click.command(help='Animate a particle trajectory')
# @click.option('--squeeze', '-s', is_flag=True,
#               help="Squeeze the components into one panel.")
# @click.option('--arg', type=click.STRING,
#               help="Additional plotting arguments like '*--'.")
# @click.option('-c', '--contour', is_flag=True,
#               help="Switch to contour mode.")
# @click.option('-q', '--quiver', is_flag=True,
#               help="Switch to quiver mode.")
# @click.option('-l', '--streamline', is_flag=True,
#               help="Switch to streamline mode.")
# @click.option('-d', '--diverging', is_flag=True,
#               help="Switch to diverging colormesh mode.")
# @click.option('--style',
#               help="Specify Matplotlib style file (default: Postgkyl).")
@click.option('--fix-aspect', 'fixaspect', is_flag=True,
              help="Enforce the same scaling on both axes.")
# @click.option('--logx', is_flag=True,
#               help="Set x-axis to log scale.")
# @click.option('--logy', is_flag=True,
#               help="Set y-axis to log scale.")
# @click.option('--legend/--no-legend', default=True,
#               help="Show legend.")
@click.option('--show/--no-show', default=True,
              help="Turn showing of the plot ON and OFF (default: ON).")
# #@click.option('--color', type=click.STRING,
# #              help="Set color when available.")
# @click.option('-x', '--xlabel', type=click.STRING,
#               help="Specify a x-axis label.")
# @click.option('-y', '--ylabel', type=click.STRING,
#               help="Specify a y-axis label.")
# @click.option('-t', '--title', type=click.STRING,
#               help="Specify a title label.")
@click.option('-i', '--interval', default=100,
              help="Specify the animation interval.")
# @click.option('-f', '--float', is_flag=True,
#               help="Choose min/max levels based on current frame")
@click.option('--save', is_flag=True,
              help="Save figure as PNG.")
@click.option('--velocity/--no-velocity', default=True,
              help="Plot velocity vectors.")
@click.option('--saveas', type=click.STRING, default=None,
              help="Name to save the plot as.")
# @click.option('-e', '--edgecolors', type=click.STRING,
#               help="Set color for cell edges (default: None)")
@click.option('-e', '--elevation', type=click.FLOAT,
              help="Set elevation")
@click.option('-a', '--azimuth', type=click.FLOAT,
              help="Set azimuth")
@click.option('-n', '--numframes', type=click.INT,
              help="Set number of frames for the animation")
@click.option('--xmin', type=click.FLOAT,
              help="Minimum value of the x-coordinate")
@click.option('--xmax', type=click.FLOAT,
              help="Maximum value of the x-coordinate")
@click.option('--ymin', type=click.FLOAT,
              help="Minimum value of the y-coordinate")
@click.option('--ymax', type=click.FLOAT,
              help="Maximum value of the y-coordinate")
@click.option('--zmin', type=click.FLOAT,
              help="Minimum value of the z-coordinate")
@click.option('--zmax', type=click.FLOAT,
              help="Maximum value of the z-coordinate")
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.pass_context
def trajectory(ctx, **kwargs):
  verb_print(ctx, 'Starting trajectory')
  data = ctx.obj['data']

  tags = list(data.tagIterator(kwargs['use']))
  if len(tags) > 1:
    ctx.fail(click.echo("'trajectory' supports only one 'tag', was provided {:d}".format(len(tags)), fg='red'))
  else:
    tag = tags[0]
  #end

  fig = plt.figure()
  ax = Axes3D(fig)
  kwargs['figure'] = fig
  kwargs['legend'] = False

  dat = ctx.obj['data'].getDataset(tag, 0)
  numPos = dat.get_num_cells()[0]

  jump = 1
  if kwargs['numframes'] is not None:
    jump = int(math.floor(numPos/kwargs['numframes']))
    numPos = int(kwargs['numframes'])
  #end

  anim = FuncAnimation(fig, update, numPos,
                       fargs=(ax, ctx, jump, kwargs['velocity'],
                              kwargs['xmin'], kwargs['xmax'],
                              kwargs['ymin'], kwargs['ymax'],
                              kwargs['zmin'], kwargs['zmax'],
                              tag),
                       interval=kwargs['interval'])

  ax.view_init(elev=kwargs['elevation'], azim=kwargs['azimuth'])

  if kwargs['fixaspect']:
    plt.setp(ax, aspect=1.0)
  #end

  fName = 'anim.mp4'
  if kwargs['saveas']:
    fName = str(kwargs['saveas'])
  #end
  if kwargs['save'] or kwargs['saveas']:
    anim.save(fName, writer='ffmpeg')
  #end

  if kwargs['show']:
    plt.show()
  #end
  verb_print(ctx, 'Finishing trajectory')
#end
