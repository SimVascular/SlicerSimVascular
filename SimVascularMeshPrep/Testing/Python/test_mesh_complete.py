import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from svmeshcomplete import io, mesh_complete
from svmeshcomplete.face_table import Face, FaceTable
from svmeshcomplete import testing as synthetic_mesh


@pytest.fixture
def packaged(tmp_path):
    mesh = synthetic_mesh.cube_mesh()
    raw = io.write_dataset(mesh, tmp_path / "geometry" / "volume_mesh.vtu")
    result = mesh_complete.write_mesh_complete(
        raw, synthetic_mesh.cube_face_table(), tmp_path / "mesh"
    )
    return result, tmp_path


def test_volume_mesh_holds_only_tetrahedra(packaged):
    result, _ = packaged
    volume = io.read_dataset(result.volume_mesh)
    types = {volume.GetCellType(i) for i in range(volume.GetNumberOfCells())}
    assert types == {vtk.VTK_TETRA}
    assert result.element_type == "Tetra"


def test_volume_mesh_carries_the_solver_arrays(packaged):
    result, _ = packaged
    volume = io.read_dataset(result.volume_mesh)
    node_ids = vtk_to_numpy(volume.GetPointData().GetArray("GlobalNodeID"))
    element_ids = vtk_to_numpy(volume.GetCellData().GetArray("GlobalElementID"))
    assert np.array_equal(node_ids, np.arange(1, volume.GetNumberOfPoints() + 1))
    assert np.array_equal(element_ids, np.arange(1, volume.GetNumberOfCells() + 1))
    assert result.number_of_nodes == volume.GetNumberOfPoints()


def test_every_face_is_written_and_the_cells_add_up(packaged):
    result, tmp_path = packaged
    assert set(result.face_surfaces) == {"wall", "cap_inlet", "cap_outlet"}
    exterior = io.read_dataset(result.exterior_surface)
    per_face = sum(
        io.read_dataset(path).GetNumberOfCells() for path in result.face_surfaces.values()
    )
    assert per_face == exterior.GetNumberOfCells()
    assert (tmp_path / "mesh" / "mesh-surfaces" / "cap_inlet.vtp").is_file()


def test_face_node_ids_point_into_the_volume_mesh(packaged):
    """What the solver relies on: a face's GlobalNodeIDs are volume nodes, at the same place."""
    result, _ = packaged
    volume = io.read_dataset(result.volume_mesh)
    cap = io.read_dataset(result.face_surfaces["cap_inlet"])
    node_ids = vtk_to_numpy(cap.GetPointData().GetArray("GlobalNodeID"))
    assert node_ids.min() >= 1 and node_ids.max() <= volume.GetNumberOfPoints()
    for index, node_id in enumerate(node_ids):
        assert cap.GetPoint(index) == pytest.approx(volume.GetPoint(int(node_id) - 1))


def test_face_element_ids_are_the_elements_behind_the_cells(packaged):
    result, _ = packaged
    volume = io.read_dataset(result.volume_mesh)
    cap = io.read_dataset(result.face_surfaces["cap_outlet"])
    element_ids = vtk_to_numpy(cap.GetCellData().GetArray("GlobalElementID"))
    node_ids = vtk_to_numpy(cap.GetPointData().GetArray("GlobalNodeID"))
    cell_points = vtk.vtkIdList()
    for cell_index, element_id in enumerate(element_ids):
        cap.GetCellPoints(cell_index, cell_points)
        face_nodes = {
            int(node_ids[cell_points.GetId(position)])
            for position in range(cell_points.GetNumberOfIds())
        }
        volume.GetCellPoints(int(element_id) - 1, cell_points)
        element_nodes = {
            cell_points.GetId(position) + 1
            for position in range(cell_points.GetNumberOfIds())
        }
        assert face_nodes <= element_nodes


def test_walls_combined_is_the_wall_faces(packaged):
    result, _ = packaged
    walls = io.read_dataset(result.walls_combined)
    wall_face = io.read_dataset(result.face_surfaces["wall"])
    assert walls.GetNumberOfCells() == wall_face.GetNumberOfCells()


def test_mixed_element_types_are_refused(tmp_path):
    """A boundary layer left as prisms would otherwise be read as an all-prism mesh."""
    mesh = synthetic_mesh.cube_mesh()
    points = vtk.vtkIdList()
    mesh.GetCellPoints(0, points)
    wedge = vtk.vtkIdList()
    wedge.SetNumberOfIds(6)
    for position in range(6):
        wedge.SetId(position, points.GetId(position % points.GetNumberOfIds()))
    mesh.InsertNextCell(vtk.VTK_WEDGE, wedge)
    face_ids = mesh.GetCellData().GetArray("CellEntityIds")
    face_ids.InsertNextValue(0)

    raw = io.write_dataset(mesh, tmp_path / "volume_mesh.vtu")
    with pytest.raises(mesh_complete.MeshCompleteError, match="Tetrahedralize"):
        mesh_complete.write_mesh_complete(
            raw, synthetic_mesh.cube_face_table(), tmp_path / "mesh"
        )


def test_a_face_the_table_does_not_name_is_refused(tmp_path):
    mesh = synthetic_mesh.cube_mesh()
    raw = io.write_dataset(mesh, tmp_path / "volume_mesh.vtu")
    short_table = FaceTable([Face(1, "wall"), Face(2, "cap_inlet")])
    with pytest.raises(mesh_complete.MeshCompleteError, match=r"faces \[3\]"):
        mesh_complete.write_mesh_complete(raw, short_table, tmp_path / "mesh")


def test_a_table_naming_a_face_the_mesh_lacks_is_refused(tmp_path):
    mesh = synthetic_mesh.cube_mesh()
    raw = io.write_dataset(mesh, tmp_path / "volume_mesh.vtu")
    long_table = FaceTable(
        [Face(1, "wall"), Face(2, "cap_inlet"), Face(3, "cap_outlet"), Face(4, "cap_ghost")]
    )
    with pytest.raises(mesh_complete.MeshCompleteError, match="cap_ghost"):
        mesh_complete.write_mesh_complete(raw, long_table, tmp_path / "mesh")


def test_the_simvascular_face_id_name_is_read_too(tmp_path):
    mesh = synthetic_mesh.cube_mesh(face_id_array_name="ModelFaceID")
    raw = io.write_dataset(mesh, tmp_path / "volume_mesh.vtu")
    result = mesh_complete.write_mesh_complete(
        raw, synthetic_mesh.cube_face_table(), tmp_path / "mesh"
    )
    assert len(result.face_surfaces) == 3


def test_a_mesh_without_face_ids_says_so(tmp_path):
    mesh = synthetic_mesh.cube_mesh()
    mesh.GetCellData().RemoveArray("CellEntityIds")
    raw = io.write_dataset(mesh, tmp_path / "volume_mesh.vtu")
    with pytest.raises(mesh_complete.MeshCompleteError, match="Face ids array"):
        mesh_complete.write_mesh_complete(
            raw, synthetic_mesh.cube_face_table(), tmp_path / "mesh"
        )
