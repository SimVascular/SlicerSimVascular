"""The `mesh/` folder svMultiPhysics reads, out of the mesh CFD Mesh Generator wrote.

That module (SlicerExtension-VMTK) returns one vtkUnstructuredGrid holding the volume
elements and the boundary cells they stand on, every cell carrying a face id -- one per
cap, one for the wall, 0 on the volume elements. This splits it into what the solver
reads: `mesh-complete.mesh.vtu`, `mesh-complete.exterior.vtp`, `walls_combined.vtp` and
one `mesh-surfaces/<name>.vtp` per face.

Two things matter and both fail quietly. The volume file must hold volume elements and
nothing else, and it must hold one element type: `vtk_xml_parser.cpp` counts cell types
and takes the *last* kind found (line, hex, quad, tet, tri, wedge) as the kind of the
whole mesh, so leftover boundary triangles are read as a triangle mesh, and prisms beside
tets as all prisms. The latter is why a boundary layer needs 'Tetrahedralize' on.

Arrays: `GlobalNodeID` (point, 1-based) numbers the volume nodes, and a face carries the
ids of the volume nodes its own points are -- that is how the solver binds a boundary
condition, so faces keep only their own points. `GlobalElementID` (cell, 1-based) numbers
the elements, and on a face it is the element behind each boundary cell. `ModelFaceID`
says which face a boundary cell is on. `ModelRegionID` is all 1 and unread by the solver,
written because every mesh folder this workflow has run carried it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svmeshcomplete import cells, io
from svmeshcomplete.face_table import FaceTable
from svmeshcomplete.faces import FaceGeometryError, find_face_id_array

# The folder's own names. `Add_mesh/Mesh_file_path` and each `Add_face/Face_file_path` in
# solver.xml point at these, so they are a contract with the solver, not preferences.
MESH_SURFACES_DIR_NAME = "mesh-surfaces"
VOLUME_MESH_NAME = "mesh-complete.mesh.vtu"
EXTERIOR_SURFACE_NAME = "mesh-complete.exterior.vtp"
WALLS_COMBINED_NAME = "walls_combined.vtp"

NODE_ID_ARRAY = "GlobalNodeID"
ELEMENT_ID_ARRAY = "GlobalElementID"
FACE_ID_ARRAY = "ModelFaceID"
REGION_ID_ARRAY = "ModelRegionID"
SOLVER_CELL_ARRAYS = (ELEMENT_ID_ARRAY, FACE_ID_ARRAY, REGION_ID_ARRAY)

# The id the volume elements carry, which is no face at all.
VOLUME_FACE_ID = 0


class MeshCompleteError(ValueError):
    """Raised when a mesh cannot be packaged for the solver."""


@dataclass
class MeshCompleteResult:
    volume_mesh: Path
    exterior_surface: Path
    walls_combined: Path
    face_surfaces: dict[str, Path]
    number_of_nodes: int
    number_of_elements: int
    element_type: str
    face_cell_counts: dict[str, int]

    def summary(self) -> str:
        width = max(len(name) for name in self.face_cell_counts)
        faces = "\n".join(
            f"  {name:<{width}}  {self.face_cell_counts[name]:>8,} cells"
            for name in sorted(self.face_cell_counts)
        )
        return (
            f"{self.number_of_elements:,} {self.element_type} elements over "
            f"{self.number_of_nodes:,} nodes\n{len(self.face_surfaces)} faces:\n{faces}"
        )


def write_mesh_complete(
    volume_mesh,
    face_table: FaceTable,
    mesh_dir: str | Path,
    *,
    face_id_array_name: str | None = None,
) -> MeshCompleteResult:
    """Package a CFD Mesh Generator mesh into the `mesh/` folder the solver reads.

    `volume_mesh` is the mesh itself, or a path to read it from.
    """
    mesh_dir = Path(mesh_dir)
    mesh = (
        volume_mesh
        if isinstance(volume_mesh, vtk.vtkDataObject)
        else io.read_dataset(volume_mesh)
    )
    if not isinstance(mesh, vtk.vtkUnstructuredGrid):
        raise MeshCompleteError(
            f"The volume mesh is a {type(mesh).__name__}, not an unstructured grid: this "
            "wants the mesh CFD Mesh Generator made, not a surface."
        )

    try:
        face_id_array_name = face_id_array_name or find_face_id_array(mesh)
    except FaceGeometryError as error:
        raise MeshCompleteError(
            f"{error} In CFD Mesh Generator that array is named by 'Face ids array', and "
            "without it there is no telling a cap from the wall."
        ) from None
    face_ids = vtk_to_numpy(mesh.GetCellData().GetArray(face_id_array_name)).astype(np.int64)

    dimensions = cells.dimensions(mesh)
    volume_cells = np.flatnonzero(dimensions == 3)
    boundary_cells = np.flatnonzero(dimensions == 2)
    if len(volume_cells) == 0:
        raise MeshCompleteError("The volume mesh has no volume elements in it.")
    if len(boundary_cells) == 0:
        raise MeshCompleteError(
            "The volume mesh carries no boundary cells, so there are no faces to bind "
            "boundary conditions to."
        )
    _check_face_ids(face_ids, volume_cells, boundary_cells, face_table)

    node_ids = np.arange(1, mesh.GetNumberOfPoints() + 1)
    cells.add_int_array(mesh, NODE_ID_ARRAY, node_ids, on_points=True)

    volume = _extract_volume_cells(mesh, volume_cells)
    element_type = _element_type_name(volume)
    element_ids = np.arange(1, volume.GetNumberOfCells() + 1, dtype=np.int32)
    cells.add_int_array(volume, NODE_ID_ARRAY, node_ids, on_points=True)
    cells.add_int_array(volume, ELEMENT_ID_ARRAY, element_ids, on_points=False)
    cells.add_int_array(volume, REGION_ID_ARRAY, np.ones(len(element_ids)), on_points=False)

    owners = _owning_element(mesh, volume_cells, boundary_cells)
    orphans = int(np.count_nonzero(owners < 0))
    if orphans:
        raise MeshCompleteError(
            f"{orphans:,} of {len(boundary_cells):,} boundary cells sit against no volume "
            "element: the mesh and the boundary it carries do not belong to each other."
        )
    exterior = cells.as_polydata(cells.extract(mesh, boundary_cells))
    cells.add_int_array(exterior, ELEMENT_ID_ARRAY, element_ids[owners], on_points=False)
    cells.add_int_array(exterior, FACE_ID_ARRAY, face_ids[boundary_cells], on_points=False)
    cells.add_int_array(exterior, REGION_ID_ARRAY, np.ones(exterior.GetNumberOfCells()), on_points=False)
    cells.keep_only(exterior.GetCellData(), SOLVER_CELL_ARRAYS)
    cells.keep_only(exterior.GetPointData(), (NODE_ID_ARRAY,))

    written = _write_all(exterior, volume, face_table, mesh_dir)
    return MeshCompleteResult(
        volume_mesh=written["volume"],
        exterior_surface=written["exterior"],
        walls_combined=written["walls"],
        face_surfaces=written["faces"],
        number_of_nodes=len(node_ids),
        number_of_elements=len(element_ids),
        element_type=element_type,
        face_cell_counts=written["counts"],
    )


def _extract_volume_cells(mesh, cell_indices):
    """The volume elements, keeping every node of the mesh in its own order.

    Faces identify their nodes by GlobalNodeID and this is what those ids index, so it
    has to carry all of them, numbered as they were when the ids were handed out.
    """
    extracted = cells.extract(mesh, cell_indices)
    original = extracted.GetPointData().GetArray("vtkOriginalPointIds")
    if original is None:
        raise MeshCompleteError("vtkExtractSelection did not report which points it kept.")
    kept = vtk_to_numpy(original).astype(np.int64)
    if len(kept) != mesh.GetNumberOfPoints():
        raise MeshCompleteError(
            f"{mesh.GetNumberOfPoints() - len(kept):,} of the mesh's "
            f"{mesh.GetNumberOfPoints():,} points belong to no volume element, so the "
            "volume file cannot carry the node ids the faces refer to."
        )

    volume = vtk.vtkUnstructuredGrid()
    volume.SetPoints(mesh.GetPoints())
    volume.Allocate(extracted.GetNumberOfCells())
    cell_points = vtk.vtkIdList()
    remapped = vtk.vtkIdList()
    for cell_index in range(extracted.GetNumberOfCells()):
        extracted.GetCellPoints(cell_index, cell_points)
        remapped.SetNumberOfIds(cell_points.GetNumberOfIds())
        for position in range(cell_points.GetNumberOfIds()):
            remapped.SetId(position, int(kept[cell_points.GetId(position)]))
        volume.InsertNextCell(extracted.GetCellType(cell_index), remapped)
    return volume


def _owning_element(mesh, volume_cells, boundary_cells):
    """For each boundary cell, its position in `volume_cells`, or -1.

    A boundary cell's points are one face of one volume element, so the element is the
    single volume cell sharing all of them.
    """
    mesh.BuildLinks()
    position_of = {int(index): position for position, index in enumerate(volume_cells)}
    owners = np.full(len(boundary_cells), -1, dtype=np.int64)
    cell_points = vtk.vtkIdList()
    neighbours = vtk.vtkIdList()
    for position, cell_index in enumerate(boundary_cells):
        mesh.GetCellPoints(int(cell_index), cell_points)
        mesh.GetCellNeighbors(int(cell_index), cell_points, neighbours)
        for neighbour_index in range(neighbours.GetNumberOfIds()):
            neighbour = int(neighbours.GetId(neighbour_index))
            if neighbour in position_of:
                owners[position] = position_of[neighbour]
                break
    return owners


def _element_type_name(mesh) -> str:
    try:
        return cells.element_type_name(mesh)
    except ValueError as error:
        raise MeshCompleteError(
            f"The volume mesh mixes element types ({error}). svMultiPhysics reads one "
            "type per mesh and takes the last kind it counts as the kind of the whole "
            "mesh, so this would run on the wrong element rather than be refused. A "
            "boundary layer of prisms beside tetrahedra is the usual cause: turn on "
            "'Tetrahedralize' in CFD Mesh Generator and mesh again."
        ) from None


def _check_face_ids(face_ids, volume_cells, boundary_cells, face_table: FaceTable) -> None:
    """Refuse a mesh whose face ids and face table do not describe the same surface."""
    stray = set(np.unique(face_ids[volume_cells]).tolist()) - {VOLUME_FACE_ID}
    if stray:
        raise MeshCompleteError(
            f"Volume elements carry face ids {sorted(stray)} and should carry only "
            f"{VOLUME_FACE_ID}: the array read as face ids is not the one holding them."
        )
    on_boundary = set(np.unique(face_ids[boundary_cells]).tolist())
    if VOLUME_FACE_ID in on_boundary:
        count = int(np.count_nonzero(face_ids[boundary_cells] == VOLUME_FACE_ID))
        raise MeshCompleteError(
            f"{count:,} boundary cells carry face id {VOLUME_FACE_ID}, the id of no face, "
            "so they can be given no boundary condition."
        )
    tabled = set(face_table.names_by_id())
    unnamed = sorted(on_boundary - tabled)
    if unnamed:
        raise MeshCompleteError(
            f"The mesh has faces {unnamed} that the face table does not name; a cap made "
            "since the table was written is numbered above the ones it has "
            f"({sorted(tabled)}). Write the table again from the clipped surface."
        )
    missing = sorted(tabled - on_boundary)
    if missing:
        names = ", ".join(face_table.name_of(face_id) for face_id in missing)
        raise MeshCompleteError(
            f"The face table names faces {missing} ({names}) that no cell of the mesh "
            "carries: either the mesher lost a cap, or the caps have been renumbered "
            "since the table was written and the conditions would land on wrong vessels."
        )


def _write_all(exterior, volume, face_table: FaceTable, mesh_dir: Path) -> dict:
    surfaces_dir = mesh_dir / MESH_SURFACES_DIR_NAME
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "volume": io.write_dataset(volume, mesh_dir / VOLUME_MESH_NAME),
        "exterior": io.write_dataset(exterior, mesh_dir / EXTERIOR_SURFACE_NAME),
        "faces": {},
        "counts": {},
    }

    face_ids = vtk_to_numpy(exterior.GetCellData().GetArray(FACE_ID_ARRAY)).astype(np.int64)
    for face in face_table:
        surface = _face_surface(exterior, np.flatnonzero(face_ids == face.face_id))
        written["faces"][face.name] = io.write_dataset(
            surface, surfaces_dir / f"{face.name}.vtp"
        )
        written["counts"][face.name] = int(surface.GetNumberOfCells())

    wall_ids = [face.face_id for face in face_table.walls]
    walls = _face_surface(exterior, np.flatnonzero(np.isin(face_ids, wall_ids)))
    written["walls"] = io.write_dataset(walls, mesh_dir / WALLS_COMBINED_NAME)
    written["counts"]["walls_combined"] = int(walls.GetNumberOfCells())
    return written


def _face_surface(exterior, cell_indices):
    surface = cells.as_polydata(cells.extract(exterior, cell_indices))
    cells.keep_only(surface.GetCellData(), SOLVER_CELL_ARRAYS)
    cells.keep_only(surface.GetPointData(), (NODE_ID_ARRAY,))
    return surface
