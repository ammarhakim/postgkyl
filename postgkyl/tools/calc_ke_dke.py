#!/usr/bin/env python
# coding: utf-8

import postgkyl
import numpy as np
import math

def calc_ke_dke(root_file_name, initFrame, finalFrame, dim, Vol, initTime, finalTime):
    #function to calculate all the total kinetic energy and the rate of dissipation of KE
    #root_file_name is the name of the file before the numbers start
    #initFrame is the first frame and finalFrame is the final frame
    #dim gives the dimension of the simulation (2 = 2D, 3 = 3D)
    #Vol = the volume of the grid
    #returns the kinetic energy and dissipation of KE

    #calculate integrated kinetic energy
    ke = np.zeros((1,(finalFrame-initFrame+1)))
    dEk = ke
    f = postgkyl.GData(root_file_name + str(initFrame) + '.bp')
    grid = f.get_grid()
    dx = grid[0][1]-grid[0][0]
    dy = grid[1][1]-grid[1][0]
    dt = (finalTime-initTime+1)/(finalFrame-initFrame+1)
    r = 0

    if dim == 3:
        dz = grid[2][1]-grid[2][0]
    elif dim == 2:
        dz = 1

    for c in range(initFrame, finalFrame+1):
        frame = postgkyl.GData(root_file_name + "%d.bp" %c)
        data = frame.get_values()
        if dim == 2:
            rho = data[:,:,0]
            px = data[:,:,1]
            py = data[:,:,2]
            pz = data[:,:,3]
        elif dim == 3:
            rho = data[:,:,:,0]
            px = data[:,:,:,1]
            py = data[:,:,:,2]
            pz = data[:,:,:,3]

        u = px/rho
        v = py/rho
        w = pz/rho

        e = rho*(u**2 + v**2 + w**2)
        ke[0,r] = np.sum(e, axis=(0,1,2))*dx*dy*dz*Vol
        r += 1

    r = 0
    for i in range(initFrame, finalFrame-1):
        dEk[0,r] = -(ke[0,i+1]-ke[0,i])/dt
        r += 1

    return ke, dEk;

