# Postgkyl

![pytest](https://github.com/ammarhakim/postgkyl/actions/workflows/test.yml/badge.svg)

This is the Postgkyl project. It is both Python library and command-line tool
designed to provide unified access to Gkeyll data together with a broad variety
of analytical and visualization tools.

## Documentation

Full documentation of the Gkeyll project is available at
[ReadTheDocs](http://gkeyll.rtfd.io).

## Dependencies and Installation

Postgkyl requires the following packages:

  * adios2
  * click
  * matplotlib
  * msgpack-python
  * numpy
  * pytest
  * scipy
  * sympy
  * tables

We recommend creating a virtual environment and installing the dependencies
through [conda](https://conda.io/miniconda.html):
```bash
conda env create -f environment.yaml
```

The environment is then activated with
```bash
conda activate pgkyl
```

However, one can also attempt to install the dependencies directly to current
conda environment using:
```bash
conda install --file requirements.txt
```

With all the dependencies installed, both the library and the command line tool
`pgkyl` can then be installed from the source with `pip`:
```bash
pip install -e .
```

To deactivate the environment, use
```bash
conda deactivate
```

## Testing

Postgkyl utilizes [pytest](https://docs.pytest.org/) for testing. The tests can
be called manually from the root Postgkyl directory simply by using:
```
pytest [-v]
```

## License

See [Gkyl License](http://gkyl.readthedocs.io/en/latest/license.html)
for usage conditions.