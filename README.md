# About

This is the PostGkyl project. It is the Python module to provide a
unified access to Gkeyll and Gkyl data together with a broad variety of
analytical tools.

# Dependencies and Installation

Postgkyl requires the following packages:

 * numpy (1.11+)
 * scipy
 * matplotlib (2.0+)
 * pytables

You can install postgkyl directly through Conda (all dependencies will
be downloaded and installed automatically):

~~~~~~~
conda install -c pcagas postgkyl
~~~~~~~

Conda package manager can be obtained ether through the full
[Anaconda](https://www.continuum.io/downloads) distribution or the
lightweight [Miniconda](https://conda.io/miniconda.html)

# Changelog

* 2017/04/04
       - `project` method of the Interp classes now returns only
         `numpy` array of 1D coordinate fields instead of
         meshgrids. Due to the matrix-like indexing of the `values`,
         transposition is necessary when plotting with
         `pcolormesh`. E.g.:

         `plt.pcolormesh(coords[0], coords[1], values.transpose())`

       - `fixCoordSlice()` now has a new keyword `mode` that allows to
         select the input (either directly index or the value)

# License

Postgkyl uses the BSD license. The full license is available [here](LICENSE).

