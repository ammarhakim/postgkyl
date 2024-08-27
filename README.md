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

* [click](https://pypi.org/project/click/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [msgpack](https://pypi.org/project/msgpack/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [sympy](https://pypi.org/project/sympy/)
* [tables](https://pypi.org/project/tables/)

Note that Posgkyl currently does not work with NumPy >= 2.0; the update is in
the works. In addition, there are two optional dependencies:

* [adios2](https://pypi.org/project/adios2/)
* [pytest](https://pypi.org/project/pytest/)

ADIOS 2 is required for reading Gkeyll 2 `bp` output files and it is not needed
when working only with `gkylzero`. [pytest](https://docs.pytest.org/en/stable/)
is required only for developers.

### Setting up virtual environment (recommended)

We strongly recommend creating a virtual Python environment for everybody
working with more than one Python project (this includes even using both
Postgkyl and Sphinx). The two recommended options are
[venv](https://docs.python.org/3/library/venv.html) and
[mamba](https://mamba.readthedocs.io/en/latest/).

With `venv`, one can create the virtual environment with:

```bash
python -m venv /path/to/new/virtual/environments/pgkyl
```

then activate it with:

| bash/zsh | `source <venv>/bin/activate`      |
| fish     | `source <venv>/bin/activate.fish` |
| csh/tcsh | `source <venv>/bin/activate.csh`  |

and deactivate with:

```bash
deactivate
```

With `mamba`, one can create the virtual environment with:

```bash
mamba create -n pgkyl
```

then activate with:

```bash
mamba activate pgkyl
```

and deactivate with:

```bash
mamba deactivate
```

Note that with `mamba`, one can also use the provided `environment.yml` file,
which also includes dependency specifications:

```bash
mamba env create -f environment.yml
```

### Installing Postgkyl

The Postgkyl itself is installed with `pip`.[^1] Developers and uses who want to
have the most up-to-date version should install Postgkyl from the source code:

```bash
git clone git@github.com:ammarhakim/postgkyl.git
cd postgkyl
pip install -e .[adios,test]
```

Alternatively, Postgkyl can be installed directly from [PyPI](https://pypi.org/project/postgkyl/):

```bash
pip install -e postgkyl[adios,test]
```

Note that ADIOS2 is not available on PyPI for Mac OSX; therefore, Mac users who
want to use it need to install the dependency from elsewhere, for example, using
the above-mentioned `mamba` and then do *not* use the `adios` tag with `pip`.

## Testing

Postgkyl utilizes [pytest](https://docs.pytest.org/) for testing. The tests can
be called manually from the root Postgkyl directory simply by using:

```bash
pytest [-v]
```

## Authors

The full list of authors can be found [here](AUTHORS.md).

## License

See [Gkyl License](http://gkyl.readthedocs.io/en/latest/license.html)
for usage conditions.

[^1]: This does *not* require any additional modifications of `PYTHONPATH`. If
    Postgkyl was used previously through `PYTHONPATH`, we strongly recommend
    removing the path to the Postgkyl repository from the variable.
