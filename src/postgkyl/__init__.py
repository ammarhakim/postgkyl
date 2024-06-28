"""
This is the postgkyl package init.
"""

from postgkyl.version import version as __version__

# import submodules
from postgkyl import data
from postgkyl import tools
from postgkyl import output
from postgkyl import utils

# import selected classes to the root
from postgkyl.data.gdata import GData
from postgkyl.data.dg import GInterpNodal
from postgkyl.data.dg import GInterpModal

# link the command line executable to the system
from postgkyl import pgkyl
