import numpy as np
import click

#sets frame in block ctx attribute using block file name
def set_frame(ctx: click.core.Context) -> list:
  """Utility function which sets data ctx frames in multiblock data situations

  This function uses gkyl's default file name output in multiblock cases to
  identify the respective frame for each loaded in data object. It assigns the correct
  frame to each data object's ctx frame attribute. It then returns a list with all the
  identified frames in ascending order.

  The motivation for this function is to allow for easy organization of multiblock data
  objects in plotting and animation.

  Args:
    ctx: click.core.context | Object
      Context from loaded data / previous commands
  Returns:
    sorted_frame_list: list
  """
      
  data = ctx.obj["data"]

  #load in file names
  files = [dat._file_name for dat in data.iterator()]

  #iterate through file names and find smallest index where file names differ, this is where the file name is
  #this is assuming that the file names are default from gkyl
  #short file is used to iterate in order to prevent indexing error
  short_file = min(files, key=len)
  num_frame_idx = np.inf
  for i in range(len(files)):
    for j in range(len(short_file)):
      if short_file[j] != files[i][j] and j < num_frame_idx:
        num_frame_idx = j
      #end
    #end
  #end

  #isolate frame number in file name and append it to big frame_list
  frame_list = []
  for f in files:
    f = f.split(".gkyl")[0]
    frame = f[num_frame_idx:].split("_")[0]
    frame_list.append(int(frame))
  #end
    
  #data objects in iterator have same index as corresponding frame in frame_list
  #this loop sets frame ctx attribute
  for i, dat in data.iterator(enum=True):
    dat.ctx["frame"] = frame_list[i]
  #end

  #return sorted frame list for use in animate function
  sorted_frame_list = np.unique(np.sort(frame_list))
  return sorted_frame_list
