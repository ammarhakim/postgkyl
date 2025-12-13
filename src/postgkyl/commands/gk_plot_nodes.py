import click
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.collections import LineCollection

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.utils.gk_utils as gku

@click.command()
@click.option("--name", "-n", required=True, type=click.STRING, default=None,
  help="Simulation name (also the file prefix, e.g. gk_sheath_1x2v_p1).")
@click.option("--path", "-p", type=click.STRING, default='./.',
  help="Path to simulation data.")
@click.option("--multib", "-m", is_flag=False, flag_value="-1", default="-10",
  help="Multiblock. Optional: pass block indices as comma-separated list or slice (start:stop:step). If no indices are given, all blocks are used.")
@click.option("--nodes_file", type=click.STRING, default=None, multiple=True,
  help="Grid nodes (.gkyl format).")
@click.option("--psi_file", type=click.STRING, default=None, multiple=True,
  help="Poloidal flux (.gkyl format).")
@click.option("--wall_file", type=click.STRING, default=None, multiple=True,
  help="Vacuum vessel wall (.csv format).")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Enforce the same scaling on both axes.")
@click.option("--xlim", default=None, type=click.STRING,
    help="Set limits for the x-coordinate (lower,upper)")
@click.option("--ylim", default=None, type=click.STRING,
    help="Set limits for the y-coordinate (lower,upper).")
@click.option("--xlabel", type=click.STRING, default="R (m)",
  help="Label for the x axis.")
@click.option("--ylabel", type=click.STRING, default="R (z)",
  help="Label for the y axis.")
@click.option("--title", type=click.STRING, default=None,
  help="Title for the figure.")
@click.option("--saveas", type=click.STRING, default=None,
  help="Name of figure file.")
@click.pass_context
def gk_plot_nodes(ctx, **kwargs):
  """
  \b
  Gyrokinetics: Plot nodes of the grid, with an option to overlay
  contours of the poloidal flux.

  \b
  The default assumes these are in the current directory.
  Alternatively, the full path to each file can be specified.

  \b
  If simulation is multiblock, and you wish to specify files manually:
    1) Pass * for the block index.
    2) Use --multib/-m to specify desired blocks (or ommit to use all).

  NOTE: this command cannot be combined with other postgkyl commands.
  """

  #
  # Hardcoded parameters and auxiliary functions.
  #

  #
  # End of hardcoded parameters and auxiliary functions.
  #

  data = ctx.obj["data"]  # Data stack.

  verb_print(ctx, "Plotting nodes for " + kwargs["name"])

  kwargs["path"] = kwargs["path"] + '/' # For safety.

  # File name root including path.
  if kwargs["multib"] == "-10":
    file_path_prefix = kwargs["path"] + kwargs["name"] + '-' # Single block.
  else:
    file_path_prefix = kwargs["path"] + kwargs["name"] + '_b*-' # Multi block.

  # File with nodes to plot.
  if kwargs["nodes_file"]:
    nodes_file = kwargs["path"] + kwargs["nodes_file"]
  else:
    nodes_file = file_path_prefix + 'nodes.gkyl'

  # Determine number of blocks.
  blocks = gku.get_block_indices(kwargs["multib"], nodes_file)
  num_blocks = len(blocks)

  block_path_prefix = file_path_prefix

  # Loop through blocks to find extrema.
  majorR_ex = [1e9, -1e9]
  vertZ_ex = [1e9, -1e9]
  for bI in range(num_blocks):
    block_path_prefix = file_path_prefix.replace("*",str(bI))

    # Load nodes.
    grid, nodes, gdat = gku.read_gfile(nodes_file.replace("*",str(bI)))

    majorR = nodes[:,:,0] # Major radius.
    vertZ = nodes[:,:,1] # Vertical location.

    majorR_ex = [min([majorR_ex[0],np.amin(majorR)]), max([majorR_ex[1],np.amax(majorR)])] 
    vertZ_ex = [min([vertZ_ex[0],np.amin(vertZ)]), max([vertZ_ex[1],np.amax(vertZ)])] 

  # Create figure.
  Rmin, Rmax = majorR_ex[0], majorR_ex[1]
  Zmin, Zmax = vertZ_ex[0], vertZ_ex[1]
  lengthR, lengthZ = Rmax-Rmin, Zmax-Zmin
  aspect_ratio = lengthR/lengthZ

  ax1aPos   = [0.88-(8.36*aspect_ratio)/(8.36*aspect_ratio+2.14), 0.08,
               (8.36*aspect_ratio)/(8.36*aspect_ratio+2.14), 0.88]
  figProp1a = (8.36*aspect_ratio+2.14, 8.36+1.14)
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = fig1a.add_axes(ax1aPos)

  # Loop through blocks to plot.
  hpl1a = list()
  for bI in range(num_blocks):

    block_path_prefix = file_path_prefix.replace("*",str(bI))
    # Load nodes.
    grid, nodes, gdat = gku.read_gfile(nodes_file.replace("*",str(bI)))
    gdat_out = GData(tag="nodes", label="nodes", ctx=gdat.ctx)

    majorR = nodes[:,:,0] # Major radius.
    vertZ = nodes[:,:,1] # Vertical location.

    # Plot each node.
    hpl1a.append(ax1a.plot(majorR,vertZ,marker=".", color="k", linestyle="none"))
    #plt.scatter(R,Z, marker=".")

    # Connect nodes with line segments.
    segs1 = np.stack((majorR,vertZ), axis=2)
    segs2 = segs1.transpose(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))

#    # Add datasets plotted to stack.
#    gdat_fdot.push(time_fdot, fdot)
#    data.add(gdat_fdot)
#
#    gdat_err = GData(tag="err", label="err", ctx=gdat_fdot.ctx)
#    gdat_err.push(time_fdot, mom_err)
#    data.add(gdat_err)

  ax1a.set_xlabel(kwargs["xlabel"],fontsize=gku.xy_label_font_size)
  ax1a.set_ylabel(kwargs["ylabel"],fontsize=gku.xy_label_font_size)
  ax1a.set_title(kwargs["title"],fontsize=gku.title_font_size)
  if kwargs["xlim"]:
    ax1a.set_xlim( float(kwargs["xlim"].split(",")[0]), float(kwargs["xlim"].split(",")[1]) )
  else:
    ax1a.set_xlim( Rmin-0.05*lengthR, Rmax+0.05*lengthR )

  if kwargs["ylim"]:
    ax1a.set_ylim( float(kwargs["ylim"].split(",")[0]), float(kwargs["ylim"].split(",")[1]) )
  else:
    ax1a.set_ylim( Zmin-0.05*lengthZ, Zmax+0.05*lengthZ )

  gku.set_tick_font_size(ax1a,gku.tick_font_size)

  if kwargs["saveas"]:
    plt.savefig(kwargs["saveas"])
  else:
    plt.show()

  verb_print(ctx, "Finishing nodes plot.")
