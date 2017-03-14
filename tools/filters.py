#!/usr/bin/env python
"""
Postgkyl sub-module for filtering
"""
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

def _clickCoords(event):
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

        cid = fig.canvas.mpl_connect('button_press_event', _clickCoords)
        plt.show()

        cutoff = ix
        print('Frequency cut-off selected: {}'.format(ix))
           
    # remove high frequency signal and return inverse FFT
    FT[freq >  cutoff] = 0
    FT[freq < -cutoff] = 0
    return numpy.fft.ifft(FT) 

def _butterLowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a

def _butterLowpassFilter(data, cutoff, fs, order=5):
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
