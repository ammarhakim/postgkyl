import click
import numpy as np

from postgkyl.data import GData
from postgkyl.utils import verb_print, set_frame

import postgkyl.data.select


@click.command()
@click.option("--z0", default=None, help="Indices for 0th coord (either int, float, or slice).")
@click.option("--z1", default=None, help="Indices for 1st coord (either int, float, or slice).")
@click.option("--z2", default=None, help="Indices for 2nd coord (either int, float, or slice).")
@click.option("--z3", default=None, help="Indices for 3rd coord (either int, float, or slice).")
@click.option("--z4", default=None, help="Indices for 4th coord (either int, float, or slice).")
@click.option("--z5", default=None, help="Indices for 5th coord (either int, float, or slice).")
@click.option("--comp", "-c", default=None,
    help="Indices for components (either int, slice, or coma-separated).")
@click.option("--use", "-u", help="Specify a 'tag' to apply to.")
@click.option("--tag", "-t", help="Optional tag for the resulting array.")
@click.option("--label", "-l", help="Custom label for the result")
@click.option("--multiblock", "-m", is_flag=True,
              help="Necessary parameter for multiblock lineouts in z0 or z1 dims")
@click.option("--multiframe", "-f", is_flag=True,
              help="Specify if performing select on multiple multiblock frames")
@click.pass_context
def select(ctx, **kwargs):
  """Subselect data from the active dataset(s).

  This command allows, for example, to choose a specific component of a multi-component
  dataset, select a index or coordinate range. Index ranges can also be specified using
  python slice notation (start:end:stride).
  """
  verb_print(ctx, "Starting select")
  data = ctx.obj["data"]

  #multiblock case
  if kwargs["multiblock"]:
    
    #set ctx frames
    frame_list = set_frame(ctx)
    #creates list of lists with blocks per frame if multiframe parameter
    #if not, then only one frame with all blocks
    if kwargs["multiframe"]:
      data_list = []
      for frame in frame_list:
        frame_data_list = [dat for dat in data.iterator(kwargs["use"]) if dat.ctx["frame"] == frame]
        data_list.append(frame_data_list)
      # end
    else:
      data_list = [list(data.iterator(kwargs["use"]))]
    # end


    for i, frame in enumerate(data_list):
    
      #establish lower bounds for x and y axis
      botlef_point = []
      for dim in [0,1]:
        botlef_point.append(min([dat.get_bounds()[0][dim] for dat in frame]))
      # end
      #find starting block for lineout coordinate
      if kwargs.get("z0"):
        for dat in frame:
          if dat.get_bounds()[0][0] <= float(kwargs["z0"]) <= dat.get_bounds()[1][0] and dat.get_bounds()[0][1] == botlef_point[1]:
            block = dat
          # end
        # end
      # end
      if kwargs.get("z1"):
        for dat in frame:
          if dat.get_bounds()[0][1] <= float(kwargs["z1"]) <= dat.get_bounds()[1][1] and dat.get_bounds()[0][0] == botlef_point[0]:
            block = dat
          # end
        # end
      # end
      #find neighboring blocks of starting block
      block.set_neighbors(frame)

      value_list = []

      #creates new grid and value list containing data from blocks which contain specified z0 coordinate
      if kwargs.get("z0"):
        grid, values = postgkyl.data.select(block,
                                            z0=kwargs["z0"],
                                            comp=kwargs["comp"])
        grid_list = grid
        for val in values[0]:
          value_list.append(val)
        # end
        while block._neighbors[1][1] is not None:
          block = block._neighbors[1][1]
          block.set_neighbors(data.iterator(kwargs["use"]))
          grid, values = postgkyl.data.select(block,
                                              z0=kwargs["z0"],
                                              comp=kwargs["comp"])
          grid_list[1] = np.append(grid_list[1], grid[1])
          for val in values[0]:
            value_list.append(val)
          # end
        # end
        grid_list[1] = np.unique(grid_list[1])
        value_list = np.array([value_list])
      # end


      #same but for z1 coordinate
      if kwargs.get("z1"):
        grid, values = postgkyl.data.select(block,
                                              z1=kwargs["z1"],
                                              comp=kwargs["comp"])
        grid_list = grid
        for val in values:
          value_list.append(val)
        # end
        while block._neighbors[0][1] is not None:
          block = block._neighbors[0][1]
          block.set_neighbors(data.iterator(kwargs["use"]))
          grid, values = postgkyl.data.select(block,
                                              z1=kwargs["z1"],
                                              comp=kwargs["comp"])
          grid_list[0] = np.append(grid_list[0], grid[0])
          for val in values:
            value_list.append(val)
          # end
        grid_list[0] = np.unique(grid_list[0])
        value_list = np.array(value_list)
      # end

      #loop through frame list and deactivate each
      for dat in frame:
        dat.deactivate()
      # end

      #create new gdata instance and push new stitched grid and values
      out = GData(tag=kwargs["tag"],
                  label=kwargs["label"],
                  comp_grid=ctx.obj["compgrid"])
      out.ctx["frame"] = i
      out.push(grid_list, value_list)
      data.add(out)
    # end


  else:
    for dat in data.iterator(kwargs["use"]):
      if kwargs["tag"]:
        out = GData(tag=kwargs["tag"], label=kwargs["label"],
            comp_grid=ctx.obj["compgrid"], ctx=dat.ctx)
        grid, values = postgkyl.data.select(dat,
            z0=kwargs["z0"], z1=kwargs["z1"], z2=kwargs["z2"], z3=kwargs["z3"],
            z4=kwargs["z4"], z5=kwargs["z5"], comp=kwargs["comp"])
        out.push(grid, values)
        data.add(out)
      else:
        postgkyl.data.select(dat, overwrite=True,
            z0=kwargs["z0"], z1=kwargs["z1"], z2=kwargs["z2"], z3=kwargs["z3"],
            z4=kwargs["z4"], z5=kwargs["z5"], comp=kwargs["comp"])
      # end
    # end
  # end
  verb_print(ctx, "Finishing select")
