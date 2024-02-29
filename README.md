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
conda env create -f environment.yaml
```

The environment is then activated with
```
conda activate pgkyl
```

Both the library and the command line tool `pgkyl` can then be easily installed:
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