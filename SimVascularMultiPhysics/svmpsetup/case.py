"""The folder an svMultiPhysics case is written into, and the checks on what was written.

```
<output folder>/
  solver.xml                  what svMultiPhysics is run on, from this folder
  mesh/                       the mesh-complete folder, as SimVascular Mesh Prep's Export writes it
    mesh-complete.mesh.vtu
    mesh-complete.exterior.vtp
    walls_combined.vtp
    mesh-surfaces/<name>.vtp  one file per face, named as Mesh Prep named it
  <cap>.flow                  each inflow waveform, in svMultiPhysics' own format
```

Every path in `solver.xml` is relative to its own folder, because that is the directory a job is
run from and svMultiPhysics resolves paths against it. The folder can be moved -- to a cluster,
say -- as a whole.

The mesh is written through `svmeshcomplete`, Mesh Prep's own package, so a 3D case set up here
and one packaged from Mesh Prep's panel or a terminal are the same folder, checks and all: one
element type, a face table that describes the mesh, no boundary cell standing against nothing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import vtk

from svmeshcomplete.face_table import Face, FaceTable
from svmeshcomplete.mesh_complete import (
    MESH_SURFACES_DIR_NAME,
    VOLUME_MESH_NAME,
    write_mesh_complete,
)
from svromsetup import boundary_conditions as bcs

from svmpsetup import settings as settings_module
from svmpsetup.solver_xml import solver_xml, write_flow_file

SOLVER_XML_NAME = "solver.xml"
MESH_DIR_NAME = "mesh"
FLOW_SUFFIX = ".flow"


class CaseError(ValueError):
    """Raised when a case cannot be written, or what was written does not hold together."""


@dataclass(frozen=True)
class CaseResult:
    solver_xml: Path
    faces: int
    caps: int
    flow_files: tuple
    elements: int


def problems(conditions_by_name, cap_names, values=None) -> list:
    """What stops these conditions and settings from making a 3D case, in words.

    The 0D panel's checks less the one about a source, which a 3D model has none of, and the
    settings' own: a value that is not one, or a fluid that does fewer iterations at most than at
    least.
    """
    by_id = dict(enumerate(cap_names))
    found = bcs.problems({index: conditions_by_name.get(name) for index, name in by_id.items()},
                         by_id, None, source_required=False)
    try:
        settings_module.resolve(values)
    except settings_module.SettingError as error:
        found.append(str(error))
    return found


def write_case(output_dir, volume_mesh, names, conditions_by_name, values=None, *,
               face_id_array_name: str | None = None) -> CaseResult:
    """Write the mesh, the inflow waveforms and `solver.xml` into `output_dir`.

    :param volume_mesh: the volume mesh CFD Mesh Generator made: a surface is refused, because a
      3D simulation is solved on the volume elements.
    :param names: `{face id: name}` for every face, as Mesh Prep names them.
    :param conditions_by_name: each cap's condition, by the cap's name, as the 0D panel shares
      them.
    :param values: settings that differ from their defaults.
    """
    if not isinstance(volume_mesh, vtk.vtkUnstructuredGrid):
        raise CaseError("A 3D simulation is solved on a volume mesh, and this is a "
                        f"{type(volume_mesh).__name__}: mesh the surface in CFD Mesh Generator "
                        "first.")
    table = FaceTable([Face(int(face_id), name) for face_id, name in names.items()])
    found = problems(conditions_by_name, [face.name for face in table.caps], values)
    if found:
        raise CaseError(" ".join(found))
    resolved = settings_module.resolve(values)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = write_mesh_complete(volume_mesh, table, output_dir / MESH_DIR_NAME,
                               face_id_array_name=face_id_array_name)

    # A renamed cap leaves its old waveform behind, and nothing would tell the stale one from a
    # live one by looking at the folder; the ones written now are the only ones there.
    for stale in output_dir.glob(f"*{FLOW_SUFFIX}"):
        stale.unlink()
    flow_files = {}
    for face in table.caps:
        condition = conditions_by_name[face.name]
        if isinstance(condition, bcs.Inflow) and (condition.source or not condition.is_steady):
            path = output_dir / f"{face.name}{FLOW_SUFFIX}"
            write_flow_file(path, condition, resolved["bc.fourier_coefficients"])
            flow_files[face.name] = path.name

    surfaces = f"{MESH_DIR_NAME}/{MESH_SURFACES_DIR_NAME}"
    text = solver_xml(
        list(table), conditions_by_name, values,
        volume_mesh_path=f"{MESH_DIR_NAME}/{VOLUME_MESH_NAME}",
        face_paths={face.name: f"{surfaces}/{face.name}.vtp" for face in table},
        flow_file_paths=flow_files)
    path = output_dir / SOLVER_XML_NAME
    path.write_text(text)
    check_case(path)
    return CaseResult(path, len(table), len(table.caps), tuple(sorted(flow_files.values())),
                      mesh.number_of_elements)


def check_case(path) -> ET.Element:
    """Hold a written `solver.xml` to the files beside it, and return it parsed.

    The solver refuses none of what this looks for until it is running on a cluster, and some of
    it not even then: a face file that is not there stops a job after its queue wait, and a face
    of the mesh that no condition names is solved as a hole in the wall.
    """
    path = Path(path)
    folder = path.parent
    root = ET.parse(path).getroot()
    found = []
    mesh = root.find("Add_mesh")
    if mesh is None or not (folder / (mesh.findtext("Mesh_file_path") or "").strip()).is_file():
        found.append("The volume mesh it names is not there.")
    faces = {face.get("name"): (face.findtext("Face_file_path") or "").strip()
             for face in (mesh.findall("Add_face") if mesh is not None else [])}
    for name, relative in faces.items():
        if not (folder / relative).is_file():
            found.append(f"The face file for {name} ({relative}) is not there.")
    surfaces = folder / MESH_DIR_NAME / MESH_SURFACES_DIR_NAME
    unlisted = sorted({file.stem for file in surfaces.glob("*.vtp")} - set(faces))
    if unlisted:
        found.append(f"Faces of the mesh that it does not list: {', '.join(unlisted)}.")
    fluid = root.find("Add_equation[@type='fluid']")
    conditions = {bc.get("name"): bc for bc in (fluid.findall("Add_BC") if fluid is not None else [])}
    unconditioned = sorted(set(faces) - set(conditions))
    if unconditioned:
        found.append(f"Faces with no condition: {', '.join(unconditioned)}.")
    unknown = sorted(set(conditions) - set(faces))
    if unknown:
        found.append(f"Conditions on faces it does not list: {', '.join(unknown)}.")
    for name, bc in conditions.items():
        waveform = (bc.findtext("Temporal_values_file_path") or "").strip()
        if waveform and not (folder / waveform).is_file():
            found.append(f"The waveform for {name} ({waveform}) is not there.")
    if found:
        raise CaseError(f"{path} does not hold together. " + " ".join(found))
    return root
