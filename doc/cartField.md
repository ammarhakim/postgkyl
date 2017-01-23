# Cartesian fields objects

The `cartField` module contains two basic classes for the data
manipulation - `CartField` and `CartFieldDG`. The later one is a child
of `CartField` that includes addition variables and methods to treat
discontinuous Galerkin data.

To construct the object, for example use:
```python
field = postgkyl.CartField('fileName.h5')
```

Note, that the `fileName` is not necessary for the initialization. The
data can be loaded afterward using:
```python
field = postgkyl.CartField()
field.load('fileName.h5')
```

When using `CartFieldDG`, the polynomial basis and the order of
polynomial approximation must be specified:
```python
field = postgkyl.CartFieldDG('serendipity', 2, 'fileName.h5')
```

## The following methods are provided for the `CartField`

`CartField.close()`
:  Close the opened HDF5 files.

`CartField.load(fileName)`
:  Load the specified data file.

`CartField.plot(comp=0, fix1=None, fix2=None, fix3=None, fix4=None, fix5=None, fix6=None)`
: Plot the field data.  Components might be
  selected using the `comp` keyword. When multiple components are
  selected `comp=(0, 1, 2)`, a corresponding number of plots will be
  produced.

: An arbitrary number of coordinates might be fixed and the
  dimensionality of thus reduced. For example, a 1X1V distribution
  function might be loaded
  
  ```python
  distfElc = postgkyl.CartField('distfElc_0.h5')
  ```
  
  1D velocity profile plot for fixed $x_i$ may be the produced with
  
  ```python
  distfElc.plot(fix1=i)
  ```

