# About

This is the PostGkyl project. It is the Python module to provide a
unified access to Gkeyll and Gkyl data together with a broad variety of
analytical tools.

# Documentation

Documentation is available at [ReadTheDocs](http://postgkyl.rtfd.io).

# Dependencies and Installation

Postgkyl requires the following packages:

 * numpy (1.11+)
 * scipy
 * matplotlib (2.0+)
 * pytables
 * click
 * adios

You can install postgkyl directly through Conda (all dependencies will
be downloaded and installed automatically):

~~~~~~~
conda install -c gkyl postgkyl
~~~~~~~

Conda package manager can be obtained ether through the full
[Anaconda](https://www.continuum.io/downloads) distribution or the
lightweight [Miniconda](https://conda.io/miniconda.html)

# Changelog

* 2017/06/22
       - Switching to 0-indexig for all commands

* 2017/05/18
       - `gplot`has been depracated. It was replaced by `pgkyl.py`
         that uses very different modular design. See the
         documentation for more details. Alternativelly, available
         commands can be listed with:

 	 `pgkyl --help`

* 2017/04/05
       - Flags `-p` and `-y` have been deprecated.

       - Plotting of multiple files/components has been added. For
         example, the following command will produced 4 lines:

         `gplot file1 file2 -c1 -c2`

       - Image output file is now set with the flag `--saveAs`. Flag
         `--save` is no longer necessary when the output file is set.

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

PostGkyl uses the BSD license. The full license is available [here](LICENSE).

# Developer guidelines

* Since Python 3 has now all the vital parts (Python 3.6 is generally
  considered the first improved version over Python 2.7), an effort
  has been made to make postpkyl Python 2/3 compatible. Please test
  your code against both versions (see conda
  [documentation](https://conda.io/docs/py2or3.html) how to maintain
  both versions). Checking `print a` to the code will result in
  revoking the repo access.

* postpkyl loosely follow the Python style conventions in PEP
  8. Python package `pep8` provides a useful
  [tool](https://pypi.python.org/pypi/pep8) to check the code. One
  exceptions the usage of camelNames instead of underscore_names.

