import numpy as np
#from postgkyl.data.data import Data

from postgkyl.modalDG.kernels import expand_1d

def interpolate(data, polyOrder=None):
    if polyOrder is None and data.polyOrder is not None:
        polyOrder = data.polyOrder
    else:
        # Something bad happened :D
        pass
    #end

    numDims = data.getNumDims()
    lower, upper = data.getBounds()
    numCells = data.getNumCells()
    
    intGrid = [np.linspace(lower[d],
                           upper[d],
                           numCells[d]*(polyOrder+1)+1)
               for d in range(numDims)]

    values = data.getValues()
    intValues = np.zeros(np.int32(numCells*(polyOrder+1)))
    intValues = intValues[..., np.newaxis]

    # 1D hardcoded for the test purposes
    intValues[0::int(polyOrder+1), 0] = expand_1d[int(polyOrder-1)](values, -2.0/3)
    intValues[1::int(polyOrder+1), 0] = expand_1d[int(polyOrder-1)](values, 0)
    intValues[2::int(polyOrder+1), 0] = expand_1d[int(polyOrder-1)](values, 2.0/3)

    # Hardcoded stack 
    data.pushGrid(intGrid)
    data.pushValues(intValues)
#end
