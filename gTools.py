#!/usr/bin/env python
"""
Postgkyl module with random useful stuff :)
"""

import numpy
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter, freqz

def rotationMatrix(vector):
    """Calculate rotation matrix

    Inputs:
    vector

    Returns:
    3x3 rotation matrix (numpy array)
    """
    rot = numpy.zeros((3, 3))
    norm = numpy.abs(vector)
    k = vector / norm # direction unit vector

    # normalization
    norm2 = numpy.sqrt(k[1]*k[1] + k[2]*k[2])
    norm3 = numpy.sqrt((k[1]*k[1] + k[2]*k[2])**2 + 
                       k[0]*k[0]*k[1]*k[1] + 
                       k[0]*k[0]*k[2]*k[2])

    rot[0, :] =  k
    rot[1, 0] =  0
    rot[1, 1] = -k[2]/norm2
    rot[1, 2] =  k[1]/norm2
    rot[2, 0] = (k[1]*k[1] + k[2]*k[2])/norm3
    rot[2, 1] = -k[0]*k[1]/norm3
    rot[2, 2] = -k[0]*k[2]/norm3

    return rot

def findNearest(array, value):
    idx = numpy.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) 
                    < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
def findNearestIdx(array, value):
    idx = numpy.searchsorted(array, value)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) 
                    < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def fixCoordSlice(coords, values,
                  fix1=None, fix2=None, fix3=None,
                  fix4=None, fix5=None, fix6=None):
    """Fix specified coordinates and decrease dimensionality

    Parameters:
    coords -- array of coordinates
    values -- array of field values
    fix1 -- fixes the first coordinate to provided index (default None)
    fix2 -- fixes the second coordinate to provided index (default None)
    fix3 -- fixes the third coordinate to provided index (default None)
    fix4 -- fixes the fourth coordinate to provided index (default None)
    fix5 -- fixes the fifth coordinate to provided index (default None)
    fix6 -- fixes the sixth coordinate to provided index (default None)

    Returns:
    coordsOut -- coordinates with decreased number of dimensions
    valuesOut -- field values with decreased number of dimensions

    Example:
    By fixing an x-index (fix1), 1X1V simulation data transforms
    to 1D velocity profile.

    Note:
    Fixing higher dimensions than available in the data has no effect.
    """
    fix = (fix1, fix2, fix3, fix4, fix5, fix6)
    coordsOut = numpy.copy(coords)
    valuesOut = numpy.copy(values)
    for i, val in reversed(list(enumerate(fix))):
        if val is not None and len(values.shape) > i:
            # turn N-D coords into correct 1D coord array
            temp = coords[i]
            coords1D = numpy.linspace(temp.min(), temp.max(),
                                      temp.shape[i])
			idx = findNearestIdx(coord1D, val)									
            # create for mask compressing
            mask = numpy.zeros(values.shape[i])
            mask[int(idx)] = 1
            # delete coordinate matrices for the fixed coordinate
            coordsOut = numpy.delete(coordsOut, i, 0)
            coordsOut = numpy.compress(mask, coordsOut, axis=i+1)  
            coordsOut = numpy.squeeze(coordsOut)

            valuesOut = numpy.compress(mask, valuesOut, axis=i) 
            valuesOut = numpy.squeeze(valuesOut)
    return coordsOut, valuesOut

def clickCoords(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    plt.close()

def fftFiltering(data, dt=1, cutoff=None):
    """Filter data using numpy FFT.

    Parameters:
    data -- numpy array of data
    dt -- set spacing of data (default: 1)
    cutoff -- set high frequency cut-off (default: None)

    Note:
    If the cutoff is not selected, and interactive figure will pop out
    that allows for the cut-off selection
    """
    N    = len(data)
    freq = numpy.fft.fftfreq(N, dt)
    FT   = numpy.fft.fft(data)

    # Get the cut-off frequency if not specified
    if cutoff is None:
        fig, ax = plt.subplots(1, 1)
        # plot just N/2 points
        ax.semilogy(freq[1:N/2], 2.0/N*numpy.abs(FT[1:N/2]))
        ax.grid()
        ax.set_xlabel('Freq')
        ax.set_ylabel('Normalized FFT')
        ax.set_title('Please, click on the plot to select cut-off frequency')
        plt.tight_layout()

        cid = fig.canvas.mpl_connect('button_press_event', clickCoords)
        plt.show()

        cutoff = ix
        print('Frequency cut-off selected: {}'.format(ix))
           
    # remove high frequency signal and return inverse FFT
    FT[freq >  cutoff] = 0
    FT[freq < -cutoff] = 0
    return numpy.fft.ifft(FT) 

def butterLowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a

def butterLowpassFilter(data, cutoff, fs, order=5):
    b, a = butterLowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butterFiltering(data, dt, cutoff):
    """Filter data using Butterworth filter

    Parameters:
    data -- numpy array of data
    dt -- set spacing of data (default: 1)
    cutoff -- set high frequency cut-off (in Hz)
    """
    order = 6
    fs = 1/dt # sample rate
    return butterLowpassFilter(data, cutoff, fs, order)
