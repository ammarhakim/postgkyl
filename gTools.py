#!/usr/bin/env python
"""
Postgkyl module with random useful stuff :)
"""

import numpy
import matplotlib.pyplot as plt
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

def clickCoords(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    plt.close()

def fftFiltering(data, dt=1, cutoff=None):
    """Filter data using numpy FFT.

    Inputs:
    data -- numpy array of data

    Key words:
    dt     -- set spacing of data (default: 1)
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

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butterFiltering(data, dt, cutoff):
    """Filter data using Butterworth filter

    Inputs:
    data -- numpy array of data

    Key words:
    dt     -- set spacing of data (default: 1)
    cutoff -- set high frequency cut-off (in Hz)
    """
    order = 6
    fs = 1/dt # sample rate
    return butter_lowpass_filter(data, cutoff, fs, order)
