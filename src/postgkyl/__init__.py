"""
This is the postgkyl package init.

isort:skip_file
"""

from .version import version as __version__

# import submodules
from . import data
from . import tools
from . import output
from . import utils
# from . import modalDG

# import selected classes to the root
from .data.gdata import GData
from .data.dg import GInterpNodal
from .data.dg import GInterpModal

# link the command line executable to the system
from . import pgkyl
