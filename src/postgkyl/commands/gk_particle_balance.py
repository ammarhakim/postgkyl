import click
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from postgkyl.data import GData
from postgkyl.utils import verb_print

@click.command()
@click.option("--name", "-n", required=True, type=click.STRING, default=None,
  help="Simulation name (also the file prefix, e.g. gk_sheath_1x2v_p1).")
@click.option("--species", "-s", required=True, type=click.STRING, default=None,
  help="Species name.")
@click.option("--path", "-p", type=click.STRING, default='./.',
  help="Path to simulation data.")
@click.option("--relative_error", "-r", is_flag=True,
  help="Plot the relative error only.")
@click.option("--multib", "-m", is_flag=False, flag_value="-1", default="-10",
  help="Multiblock. Optional: pass block indices as comma-separated list or slice (start:stop:step). If no indices are given, all blocks are used.")
@click.option("--fdot_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of change in f over a time step.")
@click.option("--source_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of the source(s).")
@click.option("--bflux_xlower_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through lower x boundary.")
@click.option("--bflux_ylower_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through lower y boundary.")
@click.option("--bflux_zlower_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through lower z boundary.")
@click.option("--bflux_xupper_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through upper x boundary.")
@click.option("--bflux_yupper_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through upper y boundary.")
@click.option("--bflux_zupper_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of boundary flux through upper z boundary.")
@click.option("--f_file", type=click.STRING, default=None, multiple=True,
  help="Integrated moments of f.")
@click.option("--dt_file", type=click.STRING, default=None,
  help="Time step.")
@click.option("--logy", is_flag=True, default=False,
  help="Logarithmic scale for y axis.")
@click.option("--absy", is_flag=True, default=False,
  help="Take absolute value of time traces.")
@click.option("--xlabel", type=click.STRING, default="Time (s)",
  help="Label for the x axis.")
@click.option("--ylabel", type=click.STRING, default=None,
  help="Label for the y axis.")
@click.option("--title", type=click.STRING, default=None,
  help="Take absolute value of time traces.")
@click.option("--indent_left", type=click.FLOAT, default=0.0,
  help="A number in the [-0.11,0.88] range by which to shift the left boundary of the plot.")
@click.option("--add_width", type=click.FLOAT, default=0.0,
  help="A number in the [-0.86,0.13] range by which to increase the width the plot.")
@click.option("--saveas", type=click.STRING, default=None,
  help="Name of figure file.")
@click.pass_context
def gk_particle_balance(ctx, **kwargs):
  """
  \b
  Gyrokinetics: Plot the particle balance of a given species.
  Requires the following files:
    ..._fdot_integrated_moms.gkyl
    ..._source_integrated_moms.gkyl
    ..._bflux_<direction><side>_integrated_HamiltonianMoments.gkyl
  where ... means <simulation_name>-<species_name>.
  The last two files above are only needed if the simulation had
  sources or non-periodic boundaries. If the relative error is
  requested, these are also needed:
    ..._integrated_moms.gkyl
    <simulation_name>-dt.gkyl

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
  max_num_blocks = 10000

  # Labels used to identify boundary flux files.
  edges = ["lower","upper"]
  dirs = ["x","y","z"]
  # Line styles.
  line_styles = ['-','--',':','-.','None','None','None','None']
  # Font sizes.
  xy_label_font_size = 17
  title_font_size = 17
  tick_font_size = 14
  legend_font_size = 14

  # Create figure.
  figProp1a = (7.5, 4.5)
  ax1aPos = [0.11+kwargs["indent_left"], 0.15, 0.87+kwargs["add_width"], 0.78]
  fig1a = plt.figure(figsize=figProp1a)
  ax1a = fig1a.add_axes(ax1aPos)

  def set_tick_font_size(axIn,fontSizeIn):
    # Set the font size of the ticks to a given size.
    axIn.tick_params(axis='both',labelsize=fontSizeIn)
    offset_txt = axIn.yaxis.get_offset_text() # Get the text object
    offset_txt.set_size(fontSizeIn) # Set the size.
    offset_txt = axIn.xaxis.get_offset_text() # Get the text object
    offset_txt.set_size(fontSizeIn) # Set the size.

  def read_gfile_if_present(file_name):
    # Check if a Gkeyll file exists. If it does, read it and return
    # its grid, data and GData object. If it doesn't, return None.
    if os.path.exists(file_name):
      pgData = GData(file_name) # Read data with pgkyl.
      time = pgData.get_grid() # Time stamps of the simulation.
      val = pgData.get_values() # Data values.
      return True, np.squeeze(time), np.squeeze(val), pgData
    else:
      verb_print(ctx, "  -> File "+file_name+" not found. Proceeding w/o it.")
      return False, None, None, None
  
  def parse_slice_string(value):
    # Parse a 'slice()' from string, like 'start:stop:step'.
    parts = value.split(':')
    # Convert parts to integers, replacing empty strings with None for slice defaults
    parsed_parts = []
    for p in parts:
      try:
        parsed_parts.append(int(p) if p else None)
      except ValueError:
        # Handle cases where the part might not be a number
        raise ValueError(f"Invalid slice part: {p}")
    # Create the slice object with the appropriate number of arguments
    return slice(*parsed_parts)

  def accumulate_or_assign(target_arr, old_arr):
    # Accumulates old_arr into target_arr if target_arr exists,
    # otherwise assign old_arr to target_arr.
    old_arr = np.asarray(old_arr) # Ensure old_arr is a numpy array.
    if target_arr is None:
      return old_arr.copy()
    else:
      target_arr += old_arr
      return target_arr

  def absy_enabled(data_in):
    # Take the absolute value of the data
    return np.abs(data_in)

  def absy_disabled(data_in):
    # Don't take the absolute value of the data
    return data_in
  #
  # End of hardcoded parameters and auxiliary functions.
  #

  data = ctx.obj["data"]  # Data stack.
  
  verb_print(ctx, "Plotting particle balance for " + kwargs["species"] + " species.")

  absy_func = absy_disabled
  if kwargs["absy"]:
    absy_func = absy_enabled

  kwargs["path"] = kwargs["path"] + '/' # For safety.

  # Determine blocks to plot, number of blocks, and set file prefix.
  if kwargs["multib"] == "-10":
    # Single block.
    file_path_prefix = kwargs["path"] + kwargs["name"] + '-'
    blocks = [0]
    num_blocks = 1
  else:
    # Multi block.
    file_path_prefix = kwargs["path"] + kwargs["name"] + '_b*-'

    if kwargs["multib"] == "-1":
      # Find and use all blocks.
      if kwargs["fdot_file"]:
        fdot_file = kwargs["path"] + kwargs["fdot_file"]
      else:
        fdot_file = file_path_prefix + kwargs["species"] + '_fdot_integrated_moms.gkyl'

      fdot_file_list = glob.glob(fdot_file)
      num_blocks = len(fdot_file_list)
      blocks = list(range(num_blocks))
    else:
      # Use specified blocks.
      if ',' in kwargs["multib"]:
        blocks = kwargs["multib"].split(",")
        num_blocks = len(blocks)
        blocks = [int(blocks[i]) for i in range(num_blocks)]
      elif ':' in kwargs["multib"]:
        slice_obj = parse_slice_string(kwargs["multib"])
        blocks = list(range(*slice_obj.indices(max_num_blocks)))
        num_blocks = len(blocks)

      else:
        raise NameError("Blocks given to --multib -m must be a comma separated list or slice.")

  block_path_prefix = file_path_prefix

  fdot = None
  src = None
  bflux_tot = None
  for bI in range(num_blocks):

    block_path_prefix = file_path_prefix.replace("*",str(bI))

    # Load change in species over a time step.
    if kwargs["fdot_file"]:
      fdot_file = kwargs["path"] + kwargs["fdot_file"].replace("*",str(bI))
    else:
      fdot_file = block_path_prefix + kwargs["species"] + '_fdot_integrated_moms.gkyl'

    _, time_fdot, fdot_pb, gdat = read_gfile_if_present(fdot_file)
    gdat_fdot = GData(tag="fdot", label="fdot", ctx=gdat.ctx)

    # Load integrated moments of the source.
    if kwargs["source_file"]:
      src_file = kwargs["path"] + kwargs["source_file"].replace("*",str(bI))
    else:
      src_file = block_path_prefix + kwargs["species"] + '_source_integrated_moms.gkyl'

    has_src, time_src, src_pb, gdat = read_gfile_if_present(src_file)
    if has_src:
      gdat_src = GData(tag="src", label="src", ctx=gdat.ctx)

    # Load particle boundary fluxes.
    nbflux = 0
    time_bflux, bflux_pb = list(), list()
    has_bflux = False
    for d in dirs:
      for e in edges:
        if kwargs["bflux_"+d+e+"_file"]:
          bflux_file = kwargs["path"] + kwargs["bflux_"+d+e+"_file"].replace("*",str(bI))
        else:
          bflux_file = block_path_prefix + kwargs["species"] + '_bflux_'+d+e+'_integrated_HamiltonianMoments.gkyl'

        has_bflux_at_boundary, time_bflux_tmp, bflux_tmp, gdat = read_gfile_if_present(bflux_file)
        if has_bflux_at_boundary:
          gdat_bflux = GData(tag="bflux", label="bflux", ctx=gdat.ctx)
          time_bflux.append(time_bflux_tmp)
          bflux_pb.append(bflux_tmp)
          has_bflux = has_bflux or has_bflux_at_boundary
          nbflux += 1

    # Select the M0 moment.
    fdot_pb = fdot_pb[:,0]
    if has_src:
      src_pb = src_pb[:,0]
    else:
      src_pb = 0.0*fdot_pb

    if has_bflux:
      for i in range(nbflux):
        bflux_pb[i] = bflux_pb[i][:,0]

    # Add boundary fluxes of all boundaries.
    if has_bflux:
      time_bflux_tot = time_bflux[0]
      bflux_tot_pb = bflux_pb[0]
      for i in range(1,nbflux):
        bflux_tot_pb += bflux_pb[i]
    else:
      bflux_tot_pb = 0.0*fdot_pb

    # Add over blocks.
    fdot = accumulate_or_assign(fdot, fdot_pb)
    src = accumulate_or_assign(src, src_pb)
    bflux_tot = accumulate_or_assign(bflux_tot, bflux_tot_pb)


  # List of handles to lines plotted, and plot a reference line at y=0.
  hpl1a = list()
  hpl1a.append(ax1a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))

  if not kwargs["relative_error"]:
    # Plot every term in the particle balance.

    src[0] = 0.0 # Set source=0 at t=0 since we don't have fdot and bflux then.

    # Compute the error.
    mom_err = src - bflux_tot - fdot
    
    # Plot.
    hpl1a.append(ax1a.plot(time_fdot, absy_func(fdot), linestyle=line_styles[0]))
    legend_strings = [r'$\dot{f}$']
    if has_src:
      hpl1a.append(ax1a.plot(time_src, absy_func(src), linestyle=line_styles[2]))
      legend_strings.append(r'$\mathcal{S}$')

    if has_bflux:
      hpl1a.append(ax1a.plot(time_bflux_tot, absy_func(-bflux_tot), linestyle=line_styles[1]))
      legend_strings.append(r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$')

    hpl1a.append(ax1a.plot(time_fdot, absy_func(mom_err), linestyle=line_styles[3]))
    legend_strings.append(r'$E_{\dot{\mathcal{N}}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f-\dot{f}$')

    ylabel_string = ""
    if kwargs["ylabel"]:
      ylabel_string = kwargs["ylabel"]

    title_string = r'Particle balance'
    if kwargs["title"]:
      title_string = kwargs["title"]

    ax1a.legend([hpl1a[i][0] for i in range(1,len(hpl1a))], legend_strings, fontsize=legend_font_size, frameon=False)

    # Add datasets plotted to stack.
    gdat_fdot.push(time_fdot, fdot)
    data.add(gdat_fdot)

    if has_src:
      gdat_src.push(time_src, src)
      data.add(gdat_src)

    if has_bflux:
      gdat_bflux.push(time_bflux, -bflux_tot)
      data.add(gdat_bflux)

    gdat_err = GData(tag="err", label="err", ctx=gdat_fdot.ctx)
    gdat_err.push(time_fdot, mom_err)
    data.add(gdat_err)

  else:
    # Plot the relative error.
  
    if kwargs["dt_file"]:
      dt_file = kwargs["path"] + kwargs["dt_file"]
    else:
      dt_file = file_path_prefix.replace("_b*","") + 'dt.gkyl'

    _, time_dt, dt, gdat = read_gfile_if_present(dt_file)
    gdat_rel_err = GData(tag="rel_err", label="rel_err", ctx=gdat.ctx)
    
    distf = None
    for bI in range(num_blocks):

      block_path_prefix = file_path_prefix.replace("*",str(bI))

      # Load integrated moments and time step.
      if kwargs["f_file"]:
        f_file = kwargs["path"] + kwargs["f_file"].replace("*",str(bI))
      else:
        f_file = block_path_prefix + kwargs["species"] + '_integrated_moms.gkyl'

      _, time_distf, distf_pb, _ = read_gfile_if_present(f_file)

      #[ Select the M0 moment.
      distf_pb = distf_pb[:,0]

      #[ Add over blocks.
      distf = accumulate_or_assign(distf, distf_pb)
    
    # Remove the t=0 data point.
    fdot = fdot[1:]
    src = src[1:]
    bflux_tot = bflux_tot[1:]
    distf = distf[1:]

    # Compute the relative error.
    mom_err = src - bflux_tot - fdot
    mom_err_norm = mom_err*dt/distf

    # Plot.
    hpl1a.append(ax1a.plot(time_dt, absy_func(mom_err_norm)))
    
    ylabel_string = r'$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$'
    if kwargs["ylabel"]:
      ylabel_string = kwargs["ylabel"]
  
    title_string = r'Relative error in particle conservation'
    if kwargs["title"]:
      title_string = kwargs["title"]

    # Add datasets plotted to stack.
    gdat_rel_err.push(time_dt, mom_err_norm)
    data.add(gdat_rel_err)

  if kwargs["logy"]:
    ax1a.set_yscale("log")

  ax1a.set_xlabel(kwargs["xlabel"],fontsize=xy_label_font_size)
  ax1a.set_ylabel(ylabel_string,fontsize=xy_label_font_size)
  ax1a.set_title(title_string,fontsize=title_font_size)
  ax1a.set_xlim( time_fdot[0], time_fdot[-1] )
  set_tick_font_size(ax1a,tick_font_size)

  if kwargs["saveas"]:
    plt.savefig(kwargs["saveas"])
  else:
    plt.show()

  verb_print(ctx, "Finishing particle balance.")
