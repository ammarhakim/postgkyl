import click
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.collections import LineCollection
from itertools import cycle
from typing import Tuple

from postgkyl.data import GData
from postgkyl.utils import verb_print
import postgkyl.utils.gk_utils as gku
import postgkyl.utils.gkeyll_enums as gkenums


def is_geo_mapc2p(gdata):
  # Determine whether the GData object, gdata, is from a simulation with MAPC2P
  # geometry. If geometry_type is missing from the metadata, default to true.
  gdata_meta = gdata.get_ctx()
  is_mapc2p = True
  if ("geometry_type" in gdata_meta):
    if gdata_meta["geometry_type"] is not None:
      mc2p_idx = gkenums.enum_key_to_idx(gkenums.gkyl_geometry_id,"GKYL_GEOMETRY_MAPC2P")
      is_mapc2p = mc2p_idx == gdata_meta["geometry_type"]
    # end
  #end
  return is_mapc2p

def nodes_to_RZ(nodes, is_mapc2p):
  # Given the nodes array with data, compute the R-Z variables.
  yidx = 0 #[ Index in the y direction to select 3D nodes at.

  nx_nod = np.shape(nodes)
  cdim = np.size(nx_nod)-1

  cart_dim = 3
  lo_idx = [[0 for d in range(cdim)] + [cd] for cd in range(cart_dim)]
  up_idx = [[nx_nod[d] for d in range(cdim)] + [cd+1] for cd in range(cart_dim)]
    
  if (cdim == 3):
    for cd in range(cart_dim):
      lo_idx[cd][1] = yidx
      up_idx[cd][1] = yidx+1
    # end
  # end

  slices = [[slice(lo_idx[cd][d], up_idx[cd][d]) for d in range(cdim+1)] for cd in range(cart_dim)]

  if is_mapc2p:
    # Nodes in Cartesian coordinates.
    cartX = [np.squeeze(nodes[tuple(slices[d])]) for d in range(cart_dim)] # X, Y, Z

    torPhi = np.arctan2(cartX[1],cartX[0]) # Toroidal angle.
    majorR = np.sqrt(np.power(cartX[0],2) + np.power(cartX[1],2)) # Major radius.
    vertZ = cartX[2] # Vertical location.
  else:
    # Nodes in R, Z, Phi coordinates.
    majorR = np.squeeze(nodes[tuple(slices[0])]) # Major radius.
    vertZ  = np.squeeze(nodes[tuple(slices[1])]) # Vertical location.
  # end

  return majorR, vertZ


@click.command()
@click.option("--name", "-n", required=True, type=click.STRING, default=None,
  help="Simulation name (also the file prefix, e.g. gk_sheath_1x2v_p1).")
@click.option("--path", "-p", type=click.STRING, default='./.',
  help="Path to simulation data.")
@click.option("--multib", "-m", type=click.STRING, is_flag=False, flag_value="-1", default="-10",
  help="Multiblock. Optional: pass block indices as comma-separated list or slice (start:stop:step). If no indices are given, all blocks are used.")
@click.option("--nodes_file", type=click.STRING, default=None, multiple=True,
  help="Grid nodes (.gkyl format).")
@click.option("--psi_file", type=click.STRING, default=None,
  help="Poloidal flux (.gkyl format).")
@click.option("--wall_file", type=click.STRING, default=None,
  help="Vacuum vessel wall (.csv format).")
@click.option("--contour", "-c", is_flag=True, help="Plot contours of psi.")
@click.option("--clevels", type=click.STRING,
    help="Specify levels for contours: comma-separated level values or start:end:nlevels.")
@click.option("--cnlevels", type=click.INT, default=11, help="Specify the number of levels for contours.")
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
@click.option("--zlabel", type=click.STRING, default=r"$\psi$",
  help="Label for the color bar.")
@click.option("--title", type=click.STRING, default=None,
  help="Title for the figure.")
@click.option("--indent_left", type=click.FLOAT, default=0.0,
  help="A number in the [-0.11,0.88] range by which to shift the left boundary of the plot.")
@click.option("--add_width", type=click.FLOAT, default=0.0,
  help="A number in the [-0.86,0.13] range by which to increase the width the plot.")
@click.option("--multib_unicolor", is_flag=True, default=False, help="Use one color for all blocks.")
@click.option("--saveas", type=click.STRING, default=None,
  help="Name of figure file.")
@click.pass_context
def gk_nodes(ctx, **kwargs):
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
  for bI in blocks:
    block_path_prefix = file_path_prefix.replace("*",str(bI))

    # Load nodes.
    grid, nodes, gdat = gku.read_gfile(nodes_file.replace("*",str(bI)))

    is_mapc2p = is_geo_mapc2p(gdat)
    majorR, vertZ = nodes_to_RZ(nodes, is_mapc2p) # Major radius and vertical location.

    majorR_ex = [min([majorR_ex[0],np.amin(majorR)]), max([majorR_ex[1],np.amax(majorR)])] 
    vertZ_ex = [min([vertZ_ex[0],np.amin(vertZ)]), max([vertZ_ex[1],np.amax(vertZ)])] 

  # Create figure.
  Rmin, Rmax = majorR_ex[0], majorR_ex[1]
  Zmin, Zmax = vertZ_ex[0], vertZ_ex[1]
  lengthR, lengthZ = Rmax-Rmin, Zmax-Zmin
  aspect_ratio = lengthR/lengthZ

  ax1aPos   = [0.82-(8.36*aspect_ratio)/(8.36*aspect_ratio+2.5)+kwargs["indent_left"], 0.08,
               (8.36*aspect_ratio)/(8.36*aspect_ratio+2.5)+kwargs["add_width"], 0.88]
  cax1aPos  = [ax1aPos[0]+ax1aPos[2]+0.01, ax1aPos[1], 0.02, ax1aPos[3]];
  figProp1a = (8.36*aspect_ratio+2.5, 8.36+1.14)
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = fig1a.add_axes(ax1aPos)

  # Color cycler for plotting each block in a different color.
  color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
  block_colors = cycle(color_list)
  if kwargs["multib_unicolor"]:
    block_colors = cycle([color_list[0]])

  # Loop through blocks to plot.
  hpl1a = list()
  hcb1a = list()
  for bI in blocks:

    block_path_prefix = file_path_prefix.replace("*",str(bI))
    # Load nodes.
    grid, nodes, gdat = gku.read_gfile(nodes_file.replace("*",str(bI)))

    is_mapc2p = is_geo_mapc2p(gdat)
    majorR, vertZ = nodes_to_RZ(nodes, is_mapc2p) # Major radius and vertical location.

    # Plot each node.
    hpl1a.append(ax1a.plot(majorR,vertZ,marker=".", color="k", linestyle="none"))

    cdim = np.size(np.shape(nodes))-1
    # Connect nodes with line segments.
    cell_color = next(block_colors)
    if (cdim == 1):
      hpl1a.append(ax1a.plot(majorR,vertZ,color=cell_color, linestyle="-"))
    else:
      segs1 = np.stack((majorR,vertZ), axis=2)
      segs2 = segs1.transpose(1,0,2)
      hpl1a.append(ax1a.add_collection(LineCollection(segs1, color=cell_color)))
      hpl1a.append(ax1a.add_collection(LineCollection(segs2, color=cell_color)))

    # Add datasets plotted to stack.
    gdat_nodes = GData(tag="nodes", label="nodes", ctx=gdat.ctx)
    gdat_nodes.push(majorR, vertZ)
    data.add(gdat_nodes)

  if kwargs["psi_file"]:
    colorbar = True
    # Plot poloidal flux.
    psi_grid, psi_values, gdat = gku.read_interp_gfile(kwargs["psi_file"], 2, 'mt')
    # Convert nodal to cell center coordinates.
    psi_grid_cc = list()
    for d in range(len(psi_grid)):
      psi_grid_cc.append(0.5*(psi_grid[d][:-1] + psi_grid[d][1:]))

    if kwargs["contour"]:
      # Contour plot.
      if kwargs["clevels"]:
        if ":" in kwargs["clevels"]:
          s = clevels.split(":")
          psi_clevels = np.linspace(float(s[0]), float(s[1]), int(s[2]))
        else:
          psi_clevels = np.array(kwargs["clevels"].split(","))
          # Filter out empty elements
          psi_clevels = np.array(list(filter(None, psi_clevels)))
      else:
        psi_clevels = kwargs["cnlevels"]

      if isinstance(psi_clevels, np.ndarray) and len(psi_clevels) == 1:
        colorbar = False

      hpl1a.append(ax1a.contour(psi_grid_cc[0], psi_grid_cc[1], psi_values.transpose(), psi_clevels))

      # Add colorbar.
      if isinstance(psi_clevels, np.ndarray):
        if np.size(psi_clevels) == 1:
          colorbar = False

    else:
      # Color plot.
      hpl1a.append(ax1a.pcolormesh(psi_grid[0], psi_grid[1], psi_values.transpose(), cmap='inferno'))

    if colorbar:
      cbar_ax1a = fig1a.add_axes(cax1aPos)
      hcb1a.append(plt.colorbar(hpl1a[-1], ax=ax1a, cax=cbar_ax1a))
      hcb1a[0].ax.tick_params(labelsize=gku.tick_font_size)
      hcb1a[0].set_label(kwargs["zlabel"], rotation=90, labelpad=0, fontsize=gku.colorbar_label_font_size)

    # Add datasets plotted to stack.
    gdat_psi = GData(tag="psi", label="psi", ctx=gdat.ctx)
    if kwargs["contour"]:
      gdat_psi.push(psi_grid_cc, psi_values.transpose())
    else:
      gdat_psi.push(psi_grid, psi_values.transpose())

    data.add(gdat_psi)

  if kwargs["wall_file"]:
    # Plot the wall.
    wall_data = np.loadtxt(open(kwargs["wall_file"]),delimiter=',')
    ax1a.plot(wall_data[:,0],wall_data[:,1],color="grey")

  ax1a.set_xlabel(kwargs["xlabel"],fontsize=gku.xy_label_font_size)
  ax1a.set_ylabel(kwargs["ylabel"],fontsize=gku.xy_label_font_size)
  ax1a.set_title(kwargs["title"],fontsize=gku.title_font_size)
  if kwargs["xlim"]:
    ax1a.set_xlim( float(kwargs["xlim"].split(",")[0]), float(kwargs["xlim"].split(",")[1]) )
#  else:
#    ax1a.set_xlim( Rmin-0.05*lengthR, Rmax+0.05*lengthR )

  if kwargs["ylim"]:
    ax1a.set_ylim( float(kwargs["ylim"].split(",")[0]), float(kwargs["ylim"].split(",")[1]) )
#  else:
#    ax1a.set_ylim( Zmin-0.05*lengthZ, Zmax+0.05*lengthZ )

  gku.set_tick_font_size(ax1a,gku.tick_font_size)

  if kwargs["saveas"]:
    plt.savefig(kwargs["saveas"])
  else:
    plt.show()

  verb_print(ctx, "Finishing nodes plot.")
