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

    nodes = [-2.0/3, 0.0, 2.0/3]
    values = data.getValues()
    intValues = np.zeros(np.int32(numCells*len(nodes)))
    intValues = intValues[..., np.newaxis]

    if numDims == 1:
        for i, x in enumerate(nodes):
            intValues[i::len(nodes), 0] = expand_1d[int(polyOrder-1)](values, x)
        #end
    elif numDims == 2:
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                intValues[i::len(nodes), j::len(nodes), 0] = expand_2d[int(polyOrder-1)](values, x, y)
            #end
        #end
    #end
        
    # Hardcoded stack 
    data.pushGrid(intGrid)
    data.pushValues(intValues)
#end
