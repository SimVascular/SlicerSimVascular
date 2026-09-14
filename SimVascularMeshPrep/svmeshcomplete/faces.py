"""What each labeled face is, measured — so that a cap can be matched to a vessel.

Clip Vessel says a face is `Outlet 14`. Which vessel that is comes from its size and
where it sits, which is what this reports: area, the diameter of the equivalent circle,
the centroid, and how flat it is. A `cap` that is not flat was not cut cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from svmeshcomplete import cells

# In priority order: the first of these the mesh carries is the one read.
#
# `CellEntityIds` is VMTK's name and `ModelFaceID` is SimVascular's. `MaterialIds` is
# neither -- it is Slicer's own material array -- and it is here last because CFD Mesh
# Generator puts it there: its Face ids array field uses the first name the *input surface*
# already carried, and its widget prepends an integer cell array it finds on an input
# carrying neither of the other two. A surface out of Clip Vessel carries MaterialIds, so
# the ids of a mesh built on one land in it, and a mesh is no less labelled for that.
#
# Reading it cannot quietly mistake a real material array for face labels: a genuine one
# has nonzero values on the volume elements, and `mesh_complete` refuses those, naming the
# cause. It is last so that a mesh carrying both is read by the name that means faces.
FACE_ID_ARRAY_NAMES = ("CellEntityIds", "ModelFaceID", "MaterialIds")


class FaceGeometryError(ValueError):
    """Raised when a dataset carries no faces to measure."""


@dataclass(frozen=True)
class FaceGeometry:
    face_id: int
    cell_count: int
    area: float
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    flatness: float
    """Largest distance of any point from the face's best-fit plane, in the input's units."""

    @property
    def effective_diameter(self) -> float:
        """Diameter of the circle of the same area."""
        return 2.0 * float(np.sqrt(self.area / np.pi))


def find_face_id_array(dataset) -> str:
    cell_data = dataset.GetCellData()
    for name in FACE_ID_ARRAY_NAMES:
        if cell_data.GetArray(name) is not None:
            return name
    carried = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
    raise FaceGeometryError(
        f"Carries none of {list(FACE_ID_ARRAY_NAMES)} as face ids, only {carried}."
    )


def boundary_of(dataset):
    """The 2D cells of a dataset -- a volume mesh's boundary, or a surface as it is."""
    if isinstance(dataset, vtk.vtkPolyData):
        return dataset
    two_dimensional = np.flatnonzero(cells.dimensions(dataset) == 2)
    if len(two_dimensional) == 0:
        raise FaceGeometryError(
            "No 2D cells: this mesh carries no boundary, so there are no faces on it."
        )
    return cells.as_polydata(cells.extract(dataset, two_dimensional))


def measure_faces(dataset, face_id_array_name: str | None = None) -> list[FaceGeometry]:
    """Measure every labeled face of a volume mesh or a surface."""
    surface = boundary_of(dataset)
    name = face_id_array_name or find_face_id_array(surface)
    face_ids = vtk_to_numpy(surface.GetCellData().GetArray(name)).astype(np.int64)

    sizes = vtk.vtkCellSizeFilter()
    sizes.SetInputData(surface)
    sizes.ComputeAreaOn()
    sizes.ComputeVolumeOff()
    sizes.ComputeLengthOff()
    sizes.ComputeVertexCountOff()
    sizes.Update()
    areas = vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray("Area"))

    measured = []
    for face_id in sorted(set(face_ids.tolist())):
        selected = np.flatnonzero(face_ids == face_id)
        face = cells.as_polydata(cells.extract(surface, selected))
        points = vtk_to_numpy(face.GetPoints().GetData()).astype(float)
        centroid = points.mean(axis=0)
        # Best-fit plane: the normal is the least significant right singular vector.
        _, _, basis = np.linalg.svd(points - centroid, full_matrices=False)
        normal = basis[-1]
        measured.append(
            FaceGeometry(
                face_id=int(face_id),
                cell_count=len(selected),
                area=float(areas[selected].sum()),
                centroid=tuple(float(value) for value in centroid),
                normal=tuple(float(value) for value in normal),
                flatness=float(np.abs((points - centroid) @ normal).max()),
            )
        )
    if not measured:
        raise FaceGeometryError("No labeled faces found.")
    return measured
