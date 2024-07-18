"""Postgkyl module for filtering.

Contains FFT and butter filters.
"""

from scipy.signal import butter, lfilter
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def _click_coords(event):
  global ix, iy
  ix, iy = event.xdata, event.ydata
  plt.close()


def fft_filtering(data: np.ndarray, dt: float = 1.0, cutoff: Optional[float] = None) -> np.ndarray:
  """Filter data using numpy FFT.

  Args:
    data: np.ndarray
    dt: float = 1.0
      set spacing of data
    cutoff: float
      set high frequency cut-off (default: None)

  Note:
    If the cutoff is not selected, and interactive figure will pop out
    that allows for the cut-off selection.
  """
  N = len(data)
  freq = np.fft.fftfreq(N, dt)
  FT = np.fft.fft(data)

  # Get the cut-off frequency if not specified
  if cutoff is None:
    fig, ax = plt.subplots(1, 1)
    # plot just N/2 points
    ax.semilogy(freq[1:N//2], 2.0/N*np.abs(FT[1:N//2]))
    ax.grid()
    ax.set_xlabel("Freq")
    ax.set_ylabel("Normalized FFT")
    ax.set_title("Please, click on the plot to select cut-off frequency")
    plt.tight_layout()

    cid = fig.canvas.mpl_connect("button_press_event", _click_coords)
    plt.show()

    cutoff = ix
    print(f"Frequency cut-off selected: {ix}")

  # remove high frequency signal and return inverse FFT
  FT[freq > cutoff] = 0
  FT[freq < -cutoff] = 0

  return np.fft.ifft(FT)


def _butter_lowpass(cutoff: float, fs: float, order: int = 5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype="low", analog=False)
  return b, a


def _butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5):
  b, a = _butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y


def butter_filtering(data: np.ndarray, dt: float = 1.0, cutoff: Optional[float] = None) -> np.ndarray:
  """Filter data using Butterworth filter

  Args:
    data: np.ndarray
    dt: float = 1.0
      set spacing of data
    cutoff: float
      set high frequency cut-off (default: None)
  """

  order = 6
  fs = 1 / dt  # sample rate
  return _butter_lowpass_filter(data, cutoff, fs, order)
