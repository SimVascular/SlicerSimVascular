"""The case folder a 0D simulation is set up and run in, written through `svromutils`.

`svromutils` (on PyPI; svReducedOrderModelingUtils) turns a surface whose faces carry a `ModelFaceID` into an svZeroDSolver
input file: centerlines from VMTK, split into branches, each branch a chain of vessels whose
resistance, inductance and capacitance come from the centerline's section areas, and the
caps' boundary conditions at the ends. What it reads is files -- the surface, the face files,
SimVascular's boundary condition files -- so this writes them, calls it, and checks what it
wrote.

```
case/
  mesh-complete/                the faces, as SimVascular Mesh Prep's Export writes them
    mesh-complete.exterior.vtp    the surface the centerlines are traced inside
    mesh-surfaces/<name>.vtp      one file per face, named, which is what names the outlets
  centerlines.vtp               branch-split centerlines
  inflow.flow                   the inlet's flow
  flow.dat, <name>.flow         the other inflows, prescribed at their centerline ends
  rcrt.dat, resistance.dat      the outlets
  solver_0d.json                svZeroDSolver's input
  branch_results.csv            svZeroDSolver's output, per vessel segment
  cap_results.csv               the same results per cap, where the panel exports them by default
```

The face files are the same `mesh-complete` folder a 3D case of the same mesh uses, so a 0D
and a 3D simulation of one anatomy name every face the same way, and the 0D one can be kept
beside the other.

`svromutils` is imported where it is called rather than at the top, because it imports VMTK and
VMTK is not a dependency of anything else here: under Slicer it is SlicerVMTK's, and the panel
has to make it importable first.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from svmeshcomplete import cells, io
from svmeshcomplete.face_table import Face, FaceTable
from svmeshcomplete.faces import find_face_id_array
from svmeshcomplete.mesh_complete import (
    EXTERIOR_SURFACE_NAME,
    FACE_ID_ARRAY,
    MESH_SURFACES_DIR_NAME,
    write_mesh_complete,
)

from svromsetup import boundary_conditions as bcs

MESH_COMPLETE_DIR_NAME = "mesh-complete"
CENTERLINES_NAME = "centerlines.vtp"
INFLOW_NAME = "inflow.flow"
SOLVER_INPUT_NAME = "solver_0d.json"
# The two results files a case folder can hold, named for what a row is about. svZeroDSolver
# writes its results per vessel segment, `branch3_seg0`, which says nothing about which cap is
# which; the export is the same results per cap, by the names the faces go by. Called
# `results.csv` and `<model>_cap_results.csv`, side by side, they read as two versions of one
# file.
RESULTS_NAME = "branch_results.csv"
CAP_RESULTS_NAME = "cap_results.csv"

# SimVascular's names for its boundary condition files; `svromutils` picks the kind of
# condition by which of them it is given.
RCR_FILE_NAME = "rcrt.dat"
RESISTANCE_FILE_NAME = "resistance.dat"
FLOW_LIST_FILE_NAME = "flow.dat"

# The name the package gives the inlet's condition. Every other one is named after its face.
INLET_BC_NAME = "INFLOW"



class CaseError(RuntimeError):
    """Raised when a case cannot be set up, with the package's own words for why if it had any."""


@dataclass(frozen=True)
class SimulationParameters:
    """What the 0D solver is run with, in the terms its input file uses.

    Cycles and points per cycle rather than the package's time step and step count, because
    those are what `simulation_parameters` ends up saying and what anyone reading results thinks
    in. They are converted to the package's terms in `package_arguments`, and the file it
    writes is checked against them afterwards.

    `units` is the mesh's. Slicer works in millimetres and both solvers in centimetres, so the
    package scales lengths by 0.1 and areas by 0.01 for "mm"; a model in "cm" goes as it is.
    """

    cardiac_cycles: int = 5
    points_per_cycle: int = 200
    density: float = 1.06
    viscosity: float = 0.04
    units: str = "mm"

    def package_arguments(self, period: float) -> dict:
        """The package's time step and number of steps for this many cycles of this period.

        Its writer sets `number_of_time_pts_per_cardiac_cycle` to the period over the step and
        `number_of_cardiac_cycles` to one more than the steps times the step over the period,
        so the steps are one cycle short of the cycles asked for.
        """
        time_step = period / self.points_per_cycle
        return {
            "time_step": time_step,
            "num_time_steps": (self.cardiac_cycles - 1) * self.points_per_cycle,
            "density": self.density,
            "viscosity": self.viscosity,
            "units": self.units,
        }


@dataclass(frozen=True)
class Centerlines:
    geometry: object
    """The branch-split centerlines, in the mesh's own coordinates."""
    outlet_names: tuple
    """The outlet faces in the order the centerline ends are numbered, as the package paired them."""


# -- the faces ------------------------------------------------------------------
def write_surfaces(mesh, names, case_dir, face_id_array_name: str | None = None) -> Path:
    """Write the surface and one file per named face into `case/mesh-complete`.

    `mesh` is the volume mesh SimVascular Mesh Prep names -- written through its own
    `svmeshcomplete`, checks and all, so a case set up here holds the folder Mesh Prep's Export
    would have written -- or a labelled surface, which is all a 0D model needs of the geometry
    and is written directly.

    :param names: `{face id: name}` for every face, `cap_*` and `wall`/`wall_*` as Mesh Prep
      names them. The names become the face file names, and through them the boundary
      condition names, so they are what results come back labelled with.
    """
    directory = Path(case_dir) / MESH_COMPLETE_DIR_NAME
    table = FaceTable([Face(int(face_id), name) for face_id, name in names.items()])
    if isinstance(mesh, vtk.vtkUnstructuredGrid):
        write_mesh_complete(mesh, table, directory, face_id_array_name=face_id_array_name)
    else:
        _write_surface_faces(mesh, table, directory, face_id_array_name)
    return directory


def _write_surface_faces(surface, table, directory: Path, face_id_array_name):
    surfaces_dir = directory / MESH_SURFACES_DIR_NAME
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    for stale in surfaces_dir.glob("*.vtp"):
        # A renamed face is a new file, and the old one would be read as one more cap.
        stale.unlink()
    name = face_id_array_name or find_face_id_array(surface)
    face_ids = vtk_to_numpy(surface.GetCellData().GetArray(name)).astype(np.int64)
    missing = sorted(set(np.unique(face_ids).tolist()) - set(table.names_by_id()))
    if missing:
        raise CaseError(f"The surface carries faces {missing} that have no name.")
    exterior = vtk.vtkPolyData()
    exterior.DeepCopy(surface)
    cells.add_int_array(exterior, FACE_ID_ARRAY, face_ids, on_points=False)
    cells.keep_only(exterior.GetCellData(), (FACE_ID_ARRAY,))
    io.write_dataset(exterior, directory / EXTERIOR_SURFACE_NAME)
    for face in table:
        piece = cells.as_polydata(cells.extract(exterior, np.flatnonzero(face_ids == face.face_id)))
        cells.keep_only(piece.GetCellData(), (FACE_ID_ARRAY,))
        io.write_dataset(piece, surfaces_dir / f"{face.name}.vtp")


def surface_paths(case_dir):
    directory = Path(case_dir) / MESH_COMPLETE_DIR_NAME
    return directory / EXTERIOR_SURFACE_NAME, directory / MESH_SURFACES_DIR_NAME


# -- centerlines ------------------------------------------------------------------
def compute_centerlines(case_dir, inlet_face_id: int) -> Centerlines:
    """Branch-split centerlines from the inlet to every other cap, written to the case.

    The package's own centerline step, given the face files so that it finds the caps by name
    and pairs each centerline end with its face by where it is -- see
    `Centerlines.name_outlets` there.
    """
    from svromutils.create_solver_file import compute_centerlines as package_centerlines
    from svromutils.parameters import Parameters

    surface, faces_dir = surface_paths(case_dir)
    if not surface.is_file():
        raise CaseError(f"No surface at {surface}: write the faces first.")
    parameters = Parameters()
    parameters.surface_model = str(surface)
    parameters.boundary_surfaces_dir = str(faces_dir)
    parameters.inlet_face_id = int(inlet_face_id)
    parameters.centerlines_output_file = str(Path(case_dir) / CENTERLINES_NAME)
    with _package_errors("Centerlines could not be computed") as errors:
        centerlines = package_centerlines(parameters)
    if centerlines is None or centerlines.geometry is None or not centerlines.geometry.GetNumberOfCells():
        raise CaseError(_message("Centerlines could not be computed", errors))
    return Centerlines(centerlines.geometry, tuple(centerlines.outlet_face_names or ()))


def read_centerlines(case_dir):
    path = Path(case_dir) / CENTERLINES_NAME
    return io.read_dataset(path) if path.is_file() else None


# -- boundary conditions and the solver's input ---------------------------------------
def write_boundary_conditions(case_dir, conditions_by_name, inlet_name: str) -> dict:
    """Write the conditions in SimVascular's files, and say which files the package is to read.

    The inlet's flow goes in `inflow.flow`, which is the package's own inflow; every other
    inflow in `flow.dat` and a `.flow` file of its own; RCRs in `rcrt.dat` and resistances in
    `resistance.dat`. Each kind's file is written only when some face has that kind, because
    the package takes a file being named as there being outlets of that kind.

    :return: the package's arguments for them.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for stale in [case_dir / name for name in (RCR_FILE_NAME, RESISTANCE_FILE_NAME,
                                               FLOW_LIST_FILE_NAME, INFLOW_NAME)]:
        if stale.is_file():
            stale.unlink()
    for stale in case_dir.glob("*.flow"):
        stale.unlink()

    inlet = conditions_by_name.get(inlet_name)
    if not isinstance(inlet, bcs.Inflow):
        raise CaseError(f"The inlet, {inlet_name}, has to be an inflow.")
    # The cycle is read off the inlet's file, so a steady inlet beside a pulsatile inflow is
    # spread over the pulsatile one's period rather than the other way round.
    period = bcs.cycle_period(conditions_by_name.values())
    bcs.write_flow_file(case_dir / INFLOW_NAME, bcs.with_period(inlet, period))

    others = {name: bcs.with_period(condition, period)
              for name, condition in conditions_by_name.items() if name != inlet_name}
    flows = {name: c for name, c in others.items() if isinstance(c, bcs.Inflow)}
    rcrs = {name: c for name, c in others.items() if isinstance(c, bcs.RCR)}
    resistances = {name: c for name, c in others.items() if isinstance(c, bcs.Resistance)}

    files = []
    if flows:
        lines = []
        for name, inflow in flows.items():
            bcs.write_flow_file(case_dir / f"{name}.flow", inflow)
            lines.append(f"{name} {name}.flow\n")
        (case_dir / FLOW_LIST_FILE_NAME).write_text("".join(lines))
        files.append(FLOW_LIST_FILE_NAME)
    if rcrs:
        bcs.write_rcrt(case_dir / RCR_FILE_NAME, rcrs)
        files.append(RCR_FILE_NAME)
    if resistances:
        bcs.write_resistance(case_dir / RESISTANCE_FILE_NAME, resistances)
        files.append(RESISTANCE_FILE_NAME)
    if not (rcrs or resistances):
        raise CaseError("At least one outlet has to be an RCR or a resistance: with flow "
                        "prescribed everywhere, nothing sets the pressure.")
    return {
        "inflow_input_file": str(case_dir / INFLOW_NAME),
        "outflow_bc_type": ",".join(files),
        "outflow_bc_input_file": str(case_dir),
    }


def write_solver_input(case_dir, conditions_by_name, inlet_face_id: int, inlet_name: str,
                       parameters: SimulationParameters, model_name: str) -> Path:
    """Write svZeroDSolver's input for the case, from centerlines already in it.

    The centerlines are read back rather than computed again, and named again by where their
    ends are (the package does this when it is given the surface and the inlet), so that a
    rename since they were computed does not leave the outlets paired with the old names.
    """
    from svromutils.create_solver_file import create_solver_file

    case_dir = Path(case_dir)
    centerlines = case_dir / CENTERLINES_NAME
    if not centerlines.is_file():
        raise CaseError(f"No centerlines at {centerlines}: compute them first.")
    surface, faces_dir = surface_paths(case_dir)
    files = write_boundary_conditions(case_dir, conditions_by_name, inlet_name)
    period = bcs.cycle_period(conditions_by_name.values())
    output = case_dir / SOLVER_INPUT_NAME
    if output.is_file():
        output.unlink()

    with _package_errors("The solver input could not be written") as errors:
        result = create_solver_file(
            model_order=0,
            model_name=model_name,
            output_directory=str(case_dir),
            solver_output_file=SOLVER_INPUT_NAME,
            surface_model=str(surface),
            boundary_surfaces_directory=str(faces_dir),
            inlet_face_id=int(inlet_face_id),
            centerlines_input_file=str(centerlines),
            **files,
            **parameters.package_arguments(period),
        )
    if not result or not output.is_file():
        raise CaseError(_message("The solver input could not be written", errors))
    check_solver_input(output, conditions_by_name, inlet_name, parameters)
    return output


def check_solver_input(path, conditions_by_name, inlet_name: str,
                       parameters: SimulationParameters | None = None) -> dict:
    """Hold the file the package wrote to what was asked of it, and return it.

    Every cap has to have exactly one condition in it, on exactly one vessel end, and of the kind
    that was set: a condition the package dropped would leave a vessel open, and one written
    twice would have two vessels claiming the same cap. Which vessel end that is was settled by
    position when the centerlines were named, and is not open to checking by name here.
    """
    config = json.loads(Path(path).read_text())
    referenced = {}
    for vessel in config.get("vessels", []):
        for end, bc_name in vessel.get("boundary_conditions", {}).items():
            referenced.setdefault(bc_name, []).append((vessel["vessel_name"], end))
    written = {bc["bc_name"]: bc for bc in config.get("boundary_conditions", [])}

    expected = {INLET_BC_NAME: (inlet_name, "FLOW")}
    for name, condition in conditions_by_name.items():
        if name == inlet_name:
            continue
        kind = {bcs.Inflow: "FLOW", bcs.RCR: "RCR", bcs.Resistance: "RESISTANCE"}[type(condition)]
        expected[f"{kind}_{name}"] = (name, kind)

    found = []
    for bc_name, (face, kind) in expected.items():
        if bc_name not in written:
            found.append(f"{face} has no condition in the solver input.")
        elif written[bc_name]["bc_type"] != kind:
            found.append(f"{face} is written as {written[bc_name]['bc_type']}, not {kind}.")
        if len(referenced.get(bc_name, [])) != 1:
            found.append(f"{face} is the end of {len(referenced.get(bc_name, []))} vessels, not one.")
    extra = sorted(set(written) - set(expected))
    if extra:
        found.append(f"The solver input has conditions for no cap: {', '.join(extra)}.")
    if parameters is not None:
        simulation = config.get("simulation_parameters", {})
        if (simulation.get("number_of_cardiac_cycles") != parameters.cardiac_cycles
                or simulation.get("number_of_time_pts_per_cardiac_cycle")
                != parameters.points_per_cycle):
            found.append(
                f"The solver input runs {simulation.get('number_of_cardiac_cycles')} cycles of "
                f"{simulation.get('number_of_time_pts_per_cardiac_cycle')} points, and "
                f"{parameters.cardiac_cycles} of {parameters.points_per_cycle} were asked for.")
    if found:
        raise CaseError("The solver input does not say what was set up. " + " ".join(found))
    return config


# -- the package's own words ---------------------------------------------------
class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _package_errors(what: str):
    """Collect what the package logs, and turn what it raises into a CaseError saying both.

    It reports a bad parameter by logging it and returning nothing, and a bad geometry by
    raising, so both have to be caught to say anything useful.
    """
    # The package's own logger, which is where it says why it stopped: it returns an empty string
    # rather than raising when its parameters are wrong. Asked of the package rather than named
    # here, because the name has changed with the package before (generate-1d-mesh, when it was
    # sv_rom_simulation), and a handler on a logger nothing writes to collects nothing quietly.
    from svromutils.manage import get_logger_name

    logger = logging.getLogger(get_logger_name())
    handler = _Collect()
    logger.addHandler(handler)
    try:
        yield handler.messages
    except CaseError:
        raise
    except Exception as error:
        raise CaseError(_message(what, handler.messages + [str(error)])) from error
    finally:
        logger.removeHandler(handler)


def _message(what: str, messages) -> str:
    said = [message for message in messages if message and message != "Error in parameters."]
    return f"{what}: {' '.join(said)}" if said else f"{what}."
