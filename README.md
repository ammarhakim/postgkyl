# About

This is the Postgkyl project. It is both Python library and command-line tool
designed to provide unified access to Gkeyll 1.0 and 2.0 data together with a
broad variety of analytical and visualization tools.

# Documentation

Documentation is available at [ReadTheDocs](http://gkeyll.rtfd.io).

# Dependencies and Installation

Postgkyl requires the following packages:

  * adios2
  * click
  * matplotlib
  * msgpack-python
  * numpy
  * scipy
  * sympy
  * tables

We recommend creating a virtual environment and installing the dependencies
through [conda](https://conda.io/miniconda.html):
```
conda env create -f environment.yml
```

The environment is then activated with
```
conda activate pgkyl
```

However, one can also attempt to install the dependencies directly to current
conda environment using:
```
conda install --file requirements.txt
```

With all the dependencies installed, both the library and the command line tool
`pgkyl` can then be installed from the source with `pip`:
```
pip install -e .
```

To deactivate the environment, use
```
conda deactivate
```

# License

See [Gkyl License](http://gkyl.readthedocs.io/en/latest/license.html)
for usage conditions.
