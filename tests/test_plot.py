import os
import matplotlib as mpl

import postgkyl as pg

def test_plot_pcolormesh():
    data = pg.GData(f"{os.path.dirname(__file__):s}/test_data/shock-f-ser-p1.gkyl")
    img = pg.output.plot(data, )
    assert isinstance(img, mpl.collections.QuadMesh)