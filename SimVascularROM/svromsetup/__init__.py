"""A 0D simulation of a face-labelled vascular model: boundary conditions in, results out.

The model itself -- centerlines, branches, a resistance, inductance and capacitance per vessel
segment -- is `svromutils`'s, and the solving svZeroDSolver's. This is what sits on
either side of them: the boundary conditions a person sets, written in the files the package
reads; the case folder they are written into; the solver run as a process; and its results
read back per face and mapped onto the centerlines, which is where anyone looks for them.

- `boundary_conditions`: inflows, RCRs and resistances, as typed in a table, as saved with a
  scene, and as SimVascular's own files (`.flow`, `rcrt.dat`, `resistance.dat`).
- `case`: the case folder, and the calls into `svromutils`, held to what they were asked
  for afterwards.
- `solver`: svZeroDSolver (`svzerod` on PyPI) run as a process, and its CSV read.
- `network`: the solver input as a directed graph, laid out in layers from the source, for a
  panel to draw.
- `results`: per-face flows and pressures, and the centerlines coloured by them.

Nothing here imports `slicer`. `svmeshcomplete` writes the faces, and `svromutils` --
which imports VMTK -- is imported only where it is called, so everything but the centerlines
and the solver input runs with numpy and VTK alone, and so do the tests of it.
"""

from svromsetup import boundary_conditions, case, network, results, solver
from svromsetup.boundary_conditions import (
    INFLOW,
    KINDS,
    MMHG,
    RCR,
    RCR_KIND,
    RESISTANCE_KIND,
    BoundaryConditionError,
    Inflow,
    Resistance,
)
from svromsetup.case import CaseError, SimulationParameters
from svromsetup.results import FaceResult, centerline_results, face_results, write_face_results_csv
from svromsetup.solver import SolverError, find_solver, read_results, run_solver

__all__ = [
    "boundary_conditions",
    "case",
    "network",
    "results",
    "solver",
    "INFLOW",
    "KINDS",
    "MMHG",
    "RCR",
    "RCR_KIND",
    "RESISTANCE_KIND",
    "BoundaryConditionError",
    "CaseError",
    "FaceResult",
    "Inflow",
    "Resistance",
    "SimulationParameters",
    "SolverError",
    "centerline_results",
    "face_results",
    "find_solver",
    "read_results",
    "run_solver",
    "write_face_results_csv",
]
