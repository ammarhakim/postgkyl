import click

import numpy as np

from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, TYPE_CHECKING
import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import os.path
from matplotlib.collections import LineCollection

from postgkyl.utils import verb_print


@click.command()
@click.option("-u", "--use", help="Specify a 'tag' to apply to (default all tags).")
@click.option("--fix-aspect", "-a", "fixaspect", is_flag=True,
    help="Enforce the same scaling on both axes.")
@click.option("--aspect", default=None, help="Specify the scaling ratio.")
@click.pass_context
def showgrid(ctx, **kwargs):
  """Plot the grid: data file must contain nodal coordinates of the
  grid. In 2D this mean (x,y) coordinate for each node. Only 2D grids
  are supported at present.
  """

  plt.style.use(f"{os.path.dirname(os.path.realpath(__file__)):s}/../output/postgkyl.mplstyle")

  verb_print(ctx, "Starting showgrid")
  data = ctx.obj["data"]

  aspect = 'auto'
  fixaspect = False
  if kwargs["aspect"]:
    fixaspect = True
    aspect = float(kwargs["aspect"])
  # end

  if kwargs["fixaspect"]:
    fixaspect = True
    aspect = 'equal'

  for i, dat in ctx.obj["data"].iterator(kwargs["use"], enum=True):
    vals = dat.get_values()
    X = vals[:,:,0]
    Y = vals[:,:,1]
    
    plt.scatter(X, Y, s=0)

    segs1 = np.stack((X,Y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    cax = plt.gca()

    cax.add_collection(LineCollection(segs1))
    cax.add_collection(LineCollection(segs2))

    if fixaspect:
        cax.set_aspect(aspect)

    pass
  # end

  plt.show()



  verb_print(ctx, "Finishing showgrid")
