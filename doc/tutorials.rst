Tutorials
=========

This document provides basic tutorials for different parts of
Postgkyl.

gplot
-----

``gplot.py`` is a high-level command line script located in the
packages's root directory. It is designed to provide a quick acces to
data loading, discontinuous Galerkin data projections, and
plotting. Script behaviour is controled with flags.

Help containing the usage and a list of all flags with brief
descriptions can be opened with

.. code-block:: bash

   python gplot.py -h

The datafile for the base-line plotting is specified with ``--plot``
or ``-p``

.. code-block:: bash

   python gplot.py -p 
