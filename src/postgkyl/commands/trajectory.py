from matplotlib.animation import FuncAnimation
import click
import math
import matplotlib.pyplot as plt
import numpy as np

from postgkyl.utils import verb_print



def _update(i, ax, ctx, leap, vel, xmin, xmax, ymin, ymax, zmin, zmax, tag):
  colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

  s = 0
  plt.cla()
  # for s, dat in ctx.obj['data'].iterator(tag, emum=True):
  for dat in ctx.obj["data"].iterator(tag):
    time = dat.get_grid()[0]
    coords = dat.get_values()
    t_idx = int(i * leap)

    if xmin is not None:
      x = np.where(coords[:, 0] > xmin, coords[:, 0], np.nan)
    else:
      x = coords[:, 0]
    # end
    if xmax is not None:
      x = np.where(x < xmax, x, np.nan)
    # end
    if ymin is not None:
      y = np.where(coords[:, 1] > ymin, coords[:, 1], np.nan)
    else:
      y = coords[:, 1]
    # end
    if ymax is not None:
      y = np.where(y < ymax, y, np.nan)
    # end
    if zmin is not None:
      z = np.where(coords[:, 2] > zmin, coords[:, 2], np.nan)
    else:
      z = coords[:, 2]
    # end
    if zmax is not None:
      z = np.where(z < zmax, z, np.nan)
    # end

    ax.plot(x, y, z, color=colors[s % 10])
    ax.scatter(x[t_idx], y[t_idx], z[t_idx], color=colors[s % 10])
    if vel and dat.get_num_comps() == 6:
      if t_idx + leap >= len(time):
        dt = time[-1] - time[t_idx]
      else:
        dt = time[int(t_idx + leap)] - time[t_idx]
      # end
      dx = coords[i, 3] * dt
      dy = coords[i, 4] * dt
      dz = coords[i, 5] * dt
      ax.plot([x[t_idx], x[t_idx] + dx], [y[t_idx], y[t_idx] + dy], [z[t_idx], z[t_idx] + dz],
          color=colors[s % 10])
    # end
    s += 1
  # end
  plt.title(f"T: {time[t_idx]:.4e}")
  ax.set_xlabel("$z_0$")
  ax.set_ylabel("$z_1$")
  ax.set_zlabel("$z_2$")
  ax.set_xlim3d(xmin, xmax)
  ax.set_ylim3d(ymin, ymax)
  ax.set_zlim3d(zmin, zmax)


@click.command()
@click.option("--fix-aspect", "fixaspect",is_flag=True, help="Enforce the same scaling on both axes.")
@click.option("--show/--no-show", default=True, help="Turn showing of the plot ON and OFF (default: ON).")
@click.option("-i", "--interval", default=100, help="Specify the animation interval.")
@click.option("--save", is_flag=True, help="Save figure as PNG.")
@click.option("--velocity/--no-velocity", default=True, help="Plot velocity vectors.")
@click.option("--saveas", type=click.STRING, default=None, help="Name to save the plot as.")
@click.option("-e", "--elevation", type=click.FLOAT, help="Set elevation.")
@click.option("-a", "--azimuth", type=click.FLOAT, help="Set azimuth.")
@click.option("-n", "--numframes", type=click.INT, help="Set number of frames for the animation.")
@click.option("--xmin", type=click.FLOAT, help="Minimum value of the x-coordinate")
@click.option("--xmax", type=click.FLOAT, help="Maximum value of the x-coordinate")
@click.option("--ymin", type=click.FLOAT, help="Minimum value of the y-coordinate")
@click.option("--ymax", type=click.FLOAT, help="Maximum value of the y-coordinate")
@click.option("--zmin", type=click.FLOAT, help="Minimum value of the z-coordinate")
@click.option("--zmax", type=click.FLOAT, help="Maximum value of the z-coordinate")
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.pass_context
def trajectory(ctx, **kwargs):
  """Animate a particle trajectory."""
  verb_print(ctx, "Starting trajectory")
  data = ctx.obj["data"]

  tags = list(data.tag_iterator(kwargs["use"]))
  tag = tags[0]
  if len(tags) > 1:
    ctx.fail(click.echo(f"'trajectory' supports only one 'tag', was provided {len(tags):d}",
        color="red"))
  # end

  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  kwargs["figure"] = fig
  kwargs["legend"] = False

  dat = ctx.obj["data"].get_dataset(0, tag)
  num_pos = dat.get_num_cells()[0]

  jump = 1
  if kwargs["numframes"] is not None:
    jump = int(math.floor(num_pos / kwargs["numframes"]))
    num_pos = int(kwargs["numframes"])
  # end

  anim = FuncAnimation(fig, _update, num_pos,
      fargs=(ax, ctx, jump, kwargs["velocity"], kwargs["xmin"], kwargs["xmax"], kwargs["ymin"],
          kwargs["ymax"], kwargs["zmin"], kwargs["zmax"], tag),
      interval=kwargs["interval"])

  ax.view_init(elev=kwargs["elevation"], azim=kwargs["azimuth"])

  if kwargs["fixaspect"]:
    plt.setp(ax, aspect=1.0)
  # end

  f_name = "anim.mp4"
  if kwargs["saveas"]:
    f_name = str(kwargs["saveas"])
  # end
  if kwargs["save"] or kwargs["saveas"]:
    anim.save(f_name, writer="ffmpeg")
  # end

  if kwargs["show"]:
    plt.show()
  # end
  verb_print(ctx, "Finishing trajectory")
