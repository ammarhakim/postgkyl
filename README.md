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

```
conda install -c gkyl postgkyl
```

Conda package manager can be obtained ether through the full
[Anaconda](https://www.continuum.io/downloads) distribution or the
lightweight [Miniconda](https://conda.io/miniconda.html)


Note that to install a new package, users need the write permission
for the Anaconda directory. If this is not the case, one can either
create a Conda [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
or install Conda into the `$HOME` directory.

# Installing from source

If you have the full postgkyl source repository (you're likely already there if you are reading this)
you can alternatively install postgkyl from source. This allows developers to make changes to the code
and have them take effect without re-installing from conda. To install from source, the dependencies
should first be installed from conda:

```
conda install -c gkyl postgkyl --only-deps
```

Once the dependencies are installed, postgkyl can be installed by navigating into
the `postgkyl` repository and running

```
python setup.py develop
```

Note that this command only ever needs to be run once (even if one is modifying source code).
Changes to the source code will be automatically included because we have installed in
[development mode](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html)


# License

See [Gkyl License](http://gkyl.readthedocs.io/en/latest/license.html) for usage conditions.

# Developer guidelines

* Since Python 3 has now all the vital parts Postgkyl only works with
  Python 3.

* postpkyl loosely follow the Python style conventions in PEP
  8. Python package `pep8` provides a useful
  [tool](https://pypi.python.org/pypi/pep8) to check the code. One
  exceptions the usage of camelNames instead of underscore_names.

