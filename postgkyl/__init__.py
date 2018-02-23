"""Postgkyl: The Gkyl Postprocessing Package

Postgkyl is a Python post-processing tool for Gkeyll 1.0 and Gkyl
data. It allows the user to read data from HDF5 and ADIOS BP files,
manipulate the data in many wonderful ways, and then save or plot the
results. Postgkyl can be run in two modes: command line mode and
Python package mode.  See the `documentation`_ for more details.

License: 
    We follow a open-source but closed development model. Release
    zip-balls will be provided, but access to the source-code
    repository is restricted to those who need to modify the code. In
    practice, this means researchers at PPPL and our partner
    institutions. That is, those who have jointly funded projects or
    graduate students with us.

    Gkyl and Postgkyl are copyrighted 2016-2018 by Ammar Hakim.

Sub-modules:
    data: G1 and G2 data loading and handling
    diagnostics: diagnostic library
    tools: miscellaneous tools

.. _documentation:
    http://gkyl.readthedocs.io/en/latest/
"""

__version__ = '1.0'

# import submodules
from . import data
from . import diagnostics
from . import tools
from . import output

# import selected classes to the root
from .data.gdata import GData
from .data.dg import GInterpNodal
from .data.dg import GInterpModal

# link the command line executable to the system
from . import pgkyl
