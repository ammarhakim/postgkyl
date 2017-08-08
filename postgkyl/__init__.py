"""G&H project postprocessing package

Sub-modules:
 * data -- G1 and G2 data loading and handling
 * diagnostics -- various diagnostics library
 * tools -- miscellaneous useful tools
"""

# import submodules
from . import data
from . import diagnostics
from . import tools

# import selectrd classes to the root
from .data.load import GData
from .data.load import GHistoryData
from .data.dg import GInterpZeroOrder
from .data.dg import GInterpNodal
from .data.dg import GInterpModal

# link the command line executable to the system
from . import commands
from . import pgkyl
