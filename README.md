# About

This is the PostGkyl project. It is the Python module to provide a
unified access to Gkeyll 1.0 and 2.0 data together with a broad
variety of analytical tools.

# Documentation

Documentation is available at [ReadTheDocs](http://gkeyll.rtfd.io).

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

# License

See [Gkyl License](http://gkyl.readthedocs.io/en/latest/license.html) for usage conditions.

# Developer guidelines

* Since Python 3 has now all the vital parts Postgkyl only works with
  Python 3.

* postpkyl loosely follow the Python style conventions in PEP
  8. Python package `pep8` provides a useful
  [tool](https://pypi.python.org/pypi/pep8) to check the code. One
  exceptions the usage of camelNames instead of underscore_names.

