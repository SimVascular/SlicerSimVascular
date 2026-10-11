"""Put the packages on the path as Slicer does, and say where the optional parts come from.

`svromsetup` sits beside this module and `svmeshcomplete` beside SimVascular Mesh Prep, which
is how both import under Slicer: every scripted module's folder is on its path. Neither needs
installing for the tests.

`svromutils` and svZeroDSolver are not in this repository; both are on PyPI. The tests that
need `svromutils` -- the centerlines and the solver input -- run where it is installed (`pip install
svromutils`, which brings pip's VMTK with it), and are skipped elsewhere. So are the ones that run
the solver, which is `pip install svzerod`'s `svzerodsolver`, found beside this Python whether or
not its environment is activated -- or a build of its own, named by `SVZERODSOLVER`.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(MODULE_DIR.parent / "SimVascularMeshPrep"))
