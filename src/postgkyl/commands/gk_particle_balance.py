import click
import numpy as np
import matplotlib.pyplot as plt
import os

from postgkyl.data import GData
from postgkyl.utils import verb_print

@click.command()
@click.option("--name", "-n", required=True, type=click.STRING, default=None, help="Simulation name (also the file prefix, e.g. gk_sheath_1x2v_p1).")
@click.option("--species", "-s", required=True, type=click.STRING, default=None, help="Species name.")
@click.option("--path", "-p", type=click.STRING, default='./.', help="Path to simulation data.")
@click.option("--relative_error", "-r", is_flag=True, help="Plot the relative error only.")
@click.option("--fdot_file", type=click.STRING, default=None, help="Integrated moments of change in f over a time step.")
@click.option("--source_file", type=click.STRING, default=None, help="Integrated moments of the source(s).")
@click.option("--bflux_x_lower_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through lower x boundary.")
@click.option("--bflux_y_lower_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through lower y boundary.")
@click.option("--bflux_z_lower_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through lower z boundary.")
@click.option("--bflux_x_upper_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through upper x boundary.")
@click.option("--bflux_y_upper_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through upper y boundary.")
@click.option("--bflux_z_upper_file", type=click.STRING, default=None, help="Integrated moments of boundary flux through upper z boundary.")
@click.option("--f_file", type=click.STRING, default=None, help="Integrated moments of f.")
@click.option("--dt_file", type=click.STRING, default=None, help="Time step.")
@click.option("--saveas", type=click.STRING, default=None, help="Name of figure file.")
@click.pass_context
def gk_particle_balance(ctx, **kwargs):
  """
  \b
  Gyrokinetics: Plot the particle balance of a given species.
  Requires the following files:
    ... _fdot_integrated_moms.gkyl
    ... _source_integrated_moms.gkyl
    ... _bflux_<direction><side>_integrated_HamiltonianMoments.gkyl
  where ... means <simulation_name>-<species_name>.
  The last two files above are only needed if the simulation had
  sources or non-periodic boundaries. If the relative error is
  requested, these are also needed:
    ... _integrated_moms.gkyl
    <simulation_name>-dt.gkyl
  are also needed.

  The default assumes these are in the current directory.
  Alternatively, the full path to each file can be specified.

  NOTE: this command cannot be combined with other postgkyl commands.
  """
  verb_print(ctx, "Plotting particle balance for " + kwargs["species"] + " species.")

  # Labels used to identify boundary flux files.
  edges = ["lower","upper"]
  dirs = ["x","y","z"]
  # Line styles.
  line_styles = ['-','--',':','-.','None','None','None','None']
  # Font sizes.
  xy_label_font_size = 17
  tick_font_size = 14
  legend_font_size = 14

  def set_tick_font_size(axIn,fontSizeIn):
    # Set the font size of the ticks to a given size.
    axIn.tick_params(axis='both',labelsize=fontSizeIn)
    offset_txt = axIn.yaxis.get_offset_text() # Get the text object
    offset_txt.set_size(fontSizeIn) # Set the size.
    offset_txt = axIn.xaxis.get_offset_text() # Get the text object
    offset_txt.set_size(fontSizeIn) # Set the size.

  def read_dyn_vector(dataFile):
    # Read data and time stamps from a DynVector.
    pgData = GData(dataFile)  # Read data with pgkyl.
    time = pgData.get_grid()  # Time stamps of the simulation.
    val = pgData.get_values()  # Data values.
    return np.squeeze(time), np.squeeze(val)
  
  def does_file_exist(fileIn):
    # Check if a file exists.
    if os.path.exists(fileIn):
      return True
    else:
      return False
  
  kwargs["path"] = kwargs["path"] + '/' # For safety.
  file_path_prefix = kwargs["path"] + kwargs["name"] + '-'

  # Load change in species over a time step.
  if kwargs["fdot_file"]:
    fdot_file = kwargs["path"] + kwargs["fdot_file"]
  else:
    fdot_file = file_path_prefix + kwargs["species"] + '_fdot_integrated_moms.gkyl'

  time_fdot, fdot = read_dyn_vector(fdot_file)

  # Load integrated moments of the source.
  if kwargs["source_file"]:
    source_file = kwargs["path"] + kwargs["source_file"]
  else:
    source_file = file_path_prefix + kwargs["species"] + '_source_integrated_moms.gkyl'

  has_source = does_file_exist(source_file)
  if has_source:
    time_src, src = read_dyn_vector(source_file)
  else:
    verb_print(ctx, "  -> Particle source file not found.")

  # Load particle boundary fluxes.
  nbflux = 0
  time_bflux, bflux = list(), list()
  has_bflux = False
  for d in dirs:
    for e in edges:
      if kwargs["source_file"]:
        bflux_file = kwargs["path"] + kwargs["bflux_"+d+e+"_file"]
      else:
        bflux_file = file_path_prefix + kwargs["species"] + '_bflux_' + d + e + '_integrated_HamiltonianMoments.gkyl'

      has_bflux_at_boundary = does_file_exist(bflux_file)
      if has_bflux_at_boundary:
        time_bflux_tmp, bflux_tmp = read_dyn_vector(bflux_file)
        time_bflux.append(time_bflux_tmp)
        bflux.append(bflux_tmp)
        has_bflux = has_bflux or has_bflux_at_boundary
        nbflux += 1
      else:
        verb_print(ctx, "  -> File with particle fluxes through "+e+" "+d+" boundary not found.")

  # Create figure.
  figProp1a = (7.5, 4.5)
  ax1aPos = [0.11, 0.15, 0.87, 0.78]
  fig1a = plt.figure(figsize=figProp1a)
  ax1a = fig1a.add_axes(ax1aPos)

  if not kwargs["relative_error"]:
    # Plot every term in the particle balance.

    # Select the M0 moment.
    fdot = fdot[:,0]
    if has_source:
      src = src[:,0]
      src[0] = 0.0 # Set source=0 at t=0 since we don't have fdot and bflux then.
    else:
      src = 0.0*fdot

    if has_bflux:
      for i in range(nbflux):
        bflux[i] = bflux[i][:,0]
    
      time_bflux_tot = time_bflux[0]
      bflux_tot = bflux[0] # Total boundary flux loss.
      for i in range(1,nbflux):
        bflux_tot += bflux[i]
    else:
      bflux_tot = 0.0*fdot

    # Compute the error.
    mom_err = src - bflux_tot - fdot
    
    # Plot.
    hpl1a = list()
    hpl1a.append(ax1a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
    hpl1a.append(ax1a.plot(time_fdot, fdot, linestyle=line_styles[0]))
    legend_strings = [r'$\dot{f}$']
    if has_source:
      hpl1a.append(ax1a.plot(time_src, src, linestyle=line_styles[2]))
      legend_strings.append(r'$\mathcal{S}$')

    if has_bflux:
      hpl1a.append(ax1a.plot(time_bflux_tot, -bflux_tot, linestyle=line_styles[1]))
      legend_strings.append(r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$')

    hpl1a.append(ax1a.plot(time_fdot, mom_err, linestyle=line_styles[3]))
    legend_strings.append(r'$E_{\dot{\mathcal{N}}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f-\dot{f}$')

    ax1a.set_xlabel(r'Time',fontsize=xy_label_font_size)
    ax1a.set_xlim( time_fdot[0], time_fdot[-1] )
    ax1a.legend([hpl1a[i][0] for i in range(1,len(hpl1a))], legend_strings, fontsize=legend_font_size, frameon=False)
    set_tick_font_size(ax1a,tick_font_size)

  else:
    # Plot the relative error.
  
    # Load integrated moments and time step.
    if kwargs["f_file"]:
      f_file = kwargs["path"] + kwargs["f_file"]
    else:
      f_file = file_path_prefix + kwargs["species"] + '_integrated_moms.gkyl'

    time_distf, distf = read_dyn_vector(f_file)

    if kwargs["dt_file"]:
      f_file = kwargs["path"] + kwargs["dt_file"]
    else:
      f_file = file_path_prefix + 'dt.gkyl'

    time_dt, dt = read_dyn_vector(f_file)
    
    # Select the M0 moment and remove the t=0 data point.
    fdot = fdot[1:,0]
    if has_source:
      src = src[1:,0]
    else:
      src = 0.0*fdot

    if has_bflux:
      for i in range(nbflux):
        bflux[i] = bflux[i][1:,0]

      bflux_tot = bflux[0]
      for i in range(1,nbflux):
        bflux_tot += bflux[i]
    else:
      bflux_tot = 0.0*fdot

    distf = distf[1:,0]
    
    # Compute the relative error.
    mom_err = src - bflux_tot - fdot
    mom_err_norm = mom_err*dt/distf

    # Plot.
    hpl1a = list()
    hpl1a.append(ax1a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
    hpl1a.append(ax1a.plot(time_dt, mom_err_norm))
    
    ax1a.set_xlabel(r'Time',fontsize=xy_label_font_size)
    ax1a.set_ylabel(r'$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$',fontsize=xy_label_font_size)
    ax1a.set_xlim( time_fdot[0], time_fdot[-1] )
    set_tick_font_size(ax1a,tick_font_size)
    
  if kwargs["saveas"]:
    plt.savefig(kwargs["saveas"])
  else:
    plt.show()
  
  verb_print(ctx, "Finishing particle balance.")
