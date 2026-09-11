"""A stand-in for what a mesher writes, for testing a host without running one.

A cube tetrahedralized, with the boundary triangles carried alongside the tets and a face
id per cell -- 1 on the four sides (the wall), 2 on the bottom and 3 on the top (the
caps), 0 on the tets. That is the shape of a CFD Mesh Generator mesh, so it exercises the
packaging and anything built on it without Slicer, a mesher, or a case folder.

Part of the package rather than of its tests because the hosts calling it have test suites
of their own, and two copies of a fixture is two fixtures that drift.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

WALL_ID, INLET_ID, OUTLET_ID = 1, 2, 3


def cube_mesh(divisions: int = 3, face_id_array_name: str = "CellEntityIds"):
    """A tetrahedralized unit cube with its boundary cells and face ids."""
    source = vtk.vtkImageData()
    source.SetDimensions(divisions + 1, divisions + 1, divisions + 1)
    source.SetSpacing(1.0 / divisions, 1.0 / divisions, 1.0 / divisions)

    tetrahedralize = vtk.vtkDataSetTriangleFilter()
    tetrahedralize.SetInputData(source)
    tetrahedralize.Update()
    tets = tetrahedralize.GetOutput()

    boundary = vtk.vtkDataSetSurfaceFilter()
    boundary.SetInputData(tets)
    boundary.Update()
    triangles = boundary.GetOutput()

    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(tets.GetPoints())
    mesh.Allocate(tets.GetNumberOfCells() + triangles.GetNumberOfCells())
    face_ids = []

    points = vtk.vtkIdList()
    for index in range(tets.GetNumberOfCells()):
        tets.GetCellPoints(index, points)
        mesh.InsertNextCell(tets.GetCellType(index), points)
        face_ids.append(0)

    # The boundary surface's points are the mesh's own -- vtkDataSetSurfaceFilter does not
    # merge -- so its connectivity can be inserted as it is.
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(tets)
    locator.BuildLocator()
    coordinates = np.array(
        [triangles.GetPoint(index) for index in range(triangles.GetNumberOfPoints())]
    )
    mapping = [locator.FindClosestPoint(point) for point in coordinates]

    for index in range(triangles.GetNumberOfCells()):
        triangles.GetCellPoints(index, points)
        remapped = vtk.vtkIdList()
        remapped.SetNumberOfIds(points.GetNumberOfIds())
        centre = np.zeros(3)
        for position in range(points.GetNumberOfIds()):
            original = mapping[points.GetId(position)]
            remapped.SetId(position, original)
            centre += np.array(tets.GetPoint(original))
        centre /= points.GetNumberOfIds()
        mesh.InsertNextCell(vtk.VTK_TRIANGLE, remapped)
        if np.isclose(centre[2], 0.0):
            face_ids.append(INLET_ID)
        elif np.isclose(centre[2], 1.0):
            face_ids.append(OUTLET_ID)
        else:
            face_ids.append(WALL_ID)

    array = numpy_to_vtk(np.array(face_ids, dtype=np.int32), deep=True)
    array.SetName(face_id_array_name)
    mesh.GetCellData().AddArray(array)
    return mesh


def cube_face_table():
    from svmeshcomplete.face_table import Face, FaceTable

    return FaceTable(
        [
            Face(WALL_ID, "wall", "Wall"),
            Face(INLET_ID, "cap_inlet", "Inlet"),
            Face(OUTLET_ID, "cap_outlet", "Outlet 1"),
        ]
    )
