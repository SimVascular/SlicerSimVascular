"""An svMultiPhysics case for a face-labelled vascular model: the mesh, the waveforms, `solver.xml`.

What the SimVascular MultiPhysics panel writes, and what a terminal can write the same way. The
mesh is packaged by `svmeshcomplete`, as SimVascular Mesh Prep packages it; the boundary
conditions are `svromsetup`'s, the same ones the 0D panel sets, because the two panels share
them; and everything else in the file is a row of `settings`, the one table of what a rigid-wall
CFD case is run with and its defaults.

- `settings`: every setting, its default, and how a typed value is read.
- `solver_xml`: the file, built element by element from those and the conditions.
- `case`: the folder it goes in, and the checks that it refers to files that are there.

Nothing here imports `slicer`, `sv` or VMTK.
"""

from svmpsetup import case, settings, solver_xml
from svmpsetup.case import CaseError, CaseResult, check_case, problems, write_case
from svmpsetup.settings import SETTINGS, SettingError, defaults, resolve

__all__ = [
    "case",
    "settings",
    "solver_xml",
    "CaseError",
    "CaseResult",
    "SETTINGS",
    "SettingError",
    "check_case",
    "defaults",
    "problems",
    "resolve",
    "write_case",
]
