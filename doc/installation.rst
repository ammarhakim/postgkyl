************
Installation
************

.. image:: https://anaconda.org/pcagas/postgkyl/badges/version.svg
   :target: https://anaconda.org/pcagas/postgkyl
.. image:: https://anaconda.org/pcagas/postgkyl/badges/downloads.svg
   :target: https://anaconda.org/pcagas/postgkyl
.. image:: https://anaconda.org/pcagas/postgkyl/badges/installer/conda.svg
   :target: https://conda.anaconda.org/pcagas 

Postgkyl is now available through the Conda package manager from the
`Anaconda cloud <https://anaconda.org/pcagas/postgkyl>`_ for both
Python 2 and Python 3. Distribution includes packages for Linux,
Windows, and Mac OS X. Postgkyl can be installed with the single
command:

.. code-block:: bash

   conda install -c pcagas postgkyl

Another option is to permanently add the channel *pcagas* to the Conda
configuration

.. code-block:: bash

   conda config --add channels pcagas

And then install Postgkyl just with

.. code-block:: bash
		
   conda install postgkyl

.. note:: The Conda package manager is a part of both the full
	  Continuum Analytics Python distribution `Anaconda
	  <https://www.continuum.io/downloads>`_ or its light-weight version
	  `Miniconda <https://conda.io/miniconda.html>`_.

Required packages
=================

Postgkyl requires a few standard Python packages. Namely:

* numpy
* scipy
* matplotlib
* pytables

If those packages are not installed on the system, Conda will download
and install them automatically.

ADIOS package
=============

One of the few Gkyl dependencies is the parallel I/O library `ADIOS
<https://www.olcf.ornl.gov/center-projects/adios/>`_. This library has
a handy Python wrapper that can be installed using ``pip``. However,
the ADIOS itself needs to be installed first and its ``bin`` must be
on the ``PATH``

.. code-block:: bash
   
   curl http://users.nccs.gov/~pnorbert/adios-1.11.0.tar.gz > adios-1.11.0.tar.gz
   gunzip adios-1.11.0.tar.gz
   tar -xvf adios-1.11.0.tar
   cd adios-1.11.0
   CFLAGS="-fPIC" ./configure --prefix=$PREFIX --disable-fortran
   make install

.. note:: Though the ADIOS works without the flag ``-fPIC``, it is
          vital for the wrapper installation.

``pip`` installation then simply follows:
   
.. code-block:: bash

   pip install adios

.. note:: This package is only required for Gkyl output data.
