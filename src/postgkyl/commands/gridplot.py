import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines

from postgkyl.data import GData, GInterpModal
from postgkyl.utils import verb_print


@click.command()
@click.argument("name")
@click.argument("half_domain", type=bool, required=True)
@click.argument("null_type", type=click.Choice(['DN', 'SN']), required=True)
@click.argument("psisep", type=float, required=False)
@click.pass_context
def gridplot(ctx, name, half_domain, null_type, psisep):
  """Plot grid lines and nodes for a given <name>.

  Usage: pgkyl gridplot <name> [half_domain] [null_type] [psisep]
  
  Expects files like '<name>_psi.gkyl' and '<name>_bX-nodes.gkyl'.
  
  half_domain must be set to true if using an 8 block (half domain) DN geom, otherwise for full 12 block DN geoms and SN geoms half_domain must be false.
  
  Specify null_type as DN for double null configuration and SN for single null configuration
  
  """
  verb_print(ctx, "Starting gridplot")

  # Determine if configuration is double-null based on name
  DN = True if null_type == 'DN' else False

  # Load psi field and interpolate to cell centers
  psid = GData(f"{name}_psi.gkyl")
  interp = GInterpModal(psid, 2, "mt")
  grid, psi = interp.interpolate()

  # Shift grid to cell centers for contouring
  for d in range(len(grid)):
    grid[d] = 0.5 * (grid[d][:-1] + grid[d][1:])
  # end


  fig, ax = plt.subplots(figsize=(4, 9))

  # Plot separatrix
  ax.contour(grid[0], grid[1], psi[:, :, 0].transpose(),
      levels=np.r_[psisep], colors="r", linestyles="dashed")
  

  colors = ["tab:orange", "tab:blue", "tab:green", "tab:brown", "tab:purple"]
  sim_dir = "./"
  baseName = sim_dir + name
  bmin = 0
  bmax = 8 if (DN and half_domain) else (12 if DN else 6)
  simNames = [f"{baseName}_b{i}" for i in range(bmin, bmax)]
  Rlist = []
  Zlist = []
  for i, simName in enumerate(simNames):
    data = GData(simName + "-nodes.gkyl")
    vals = data.get_values()
    R = vals[:, :, 0]
    Z = vals[:, :, 1]

    # Draw grid lines and nodes
    segs1 = np.stack((R, Z), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, linewidth=0.4, color=colors[i % len(colors)]))
    ax.add_collection(LineCollection(segs2, linewidth=0.4, color=colors[i % len(colors)]))
    ax.plot(R, Z, marker=".", color="k", linestyle="none", markersize=2.0)
    
    Rlist.append(R)
    Zlist.append(Z)
    # For double null configuration, plot mirrored data with Z -> -Z
    if DN and half_domain:
      Zm = -Z
      segs1m = np.stack((R, Zm), axis=2)
      segs2m = segs1m.transpose(1, 0, 2)
      ax.add_collection(LineCollection(segs1m, linewidth=0.4, color=colors[i % len(colors)]))
      ax.add_collection(LineCollection(segs2m, linewidth=0.4, color=colors[i % len(colors)]))
      ax.plot(R, Zm, marker=".", color="k", linestyle="none", markersize=2.0)
      
      
    # end    
  # end

   #Draw divertor plates
  if DN and half_domain:
    for bidx in [4,5]:
        ax.plot(Rlist[bidx][:,-1], Zlist[bidx][:,-1], color='r', linewidth=3.0)
        ax.plot(Rlist[bidx][:,-1], -Zlist[bidx][:,-1], color='r', linewidth=3.0)


    for bidx in [0,1]:
        ax.plot(Rlist[bidx][:,0], Zlist[bidx][:,0], color='r', linewidth=3.0)
        ax.plot(Rlist[bidx][:,0], -Zlist[bidx][:,0], color='r', linewidth=3.0)

  elif DN and not half_domain:
    for bidx in [3, 4, 8, 9]:
        ax.plot(Rlist[bidx][:,-1], Zlist[bidx][:,-1], color='r', linewidth=3.0)


    for bidx in [0, 1, 5, 6]:
        ax.plot(Rlist[bidx][:,0], Zlist[bidx][:,0], color='r', linewidth=3.0)


  else: 
    for bidx in [3,4]:
      ax.plot(Rlist[bidx][:,-1], Zlist[bidx][:,-1], color='r', linewidth=3.0)

    for bidx in [0,1]:
      ax.plot(Rlist[bidx][:,0], Zlist[bidx][:,0], color='r', linewidth=3.0) 
  
  div_handle = mlines.Line2D([], [], color="r", label="Divertor", linewidth=3.0)
  sep_handle = mlines.Line2D([], [], color="r", label="Separatrix", linestyle="dashed")
  handles = [sep_handle, div_handle]
  ax.grid()
  ax.set_xlabel("R [m]")
  ax.set_ylabel("Z [m]")
  ax.axis("tight")
  ax.axis("image")
  #ax.legend(handles=handles, loc="best")

  fig.tight_layout()

  # Respect batch mode: save instead of showing
  if ctx.obj.get("batch_mode", False):
    out_prefix = ctx.obj.get("saveframes_prefix", "./pg")
    fig.savefig(f"{out_prefix}_gridplot_{name}.png", dpi=200)
  else:
    plt.show()
  # end

  verb_print(ctx, "Finishing gridplot")


