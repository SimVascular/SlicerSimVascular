"""Put the packages on the path as Slicer does: every scripted module's folder is on its path.

`svmpsetup` is beside this module, `svromsetup` beside SimVascular ROM Simulation, and
`svmeshcomplete` beside SimVascular Mesh Prep. The test that runs svMultiPhysics finds it at
`SVMULTIPHYSICS` or on the PATH, and is skipped elsewhere.
"""

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2]
for folder in (MODULE_DIR, MODULE_DIR.parent / "SimVascularROM", MODULE_DIR.parent / "SimVascularMeshPrep"):
    sys.path.insert(0, str(folder))
