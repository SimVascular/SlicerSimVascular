import pytest

from svmeshcomplete import testing as synthetic_mesh
from svmeshcomplete import faces
from svmeshcomplete.face_table import Face, FaceTable, FaceTableError


@pytest.fixture
def measured():
    return {face.face_id: face for face in faces.measure_faces(synthetic_mesh.cube_mesh())}


def test_a_face_is_measured_by_area_and_position(measured):
    # The unit cube: each cap is one 1x1 side, the wall is the other four.
    assert measured[2].area == pytest.approx(1.0)
    assert measured[1].area == pytest.approx(4.0)
    assert measured[2].effective_diameter == pytest.approx(1.128, abs=1e-3)
    assert measured[2].centroid == pytest.approx((0.5, 0.5, 0.0))


def test_a_flat_face_reads_zero_and_a_folded_one_does_not(measured):
    """Flatness is the check on a cap: a cut that was not planar cannot read 0."""
    assert measured[2].flatness == pytest.approx(0.0)
    assert measured[3].flatness == pytest.approx(0.0)
    assert measured[1].flatness > 0.1


def test_the_boundary_of_a_volume_mesh_is_its_2d_cells():
    mesh = synthetic_mesh.cube_mesh()
    surface = faces.boundary_of(mesh)
    assert surface.GetNumberOfCells() < mesh.GetNumberOfCells()
    assert faces.boundary_of(surface) is surface


def test_a_surface_with_no_face_ids_says_which_arrays_it_has():
    mesh = synthetic_mesh.cube_mesh()
    mesh.GetCellData().RemoveArray("CellEntityIds")
    with pytest.raises(faces.FaceGeometryError, match="CellEntityIds"):
        faces.measure_faces(mesh)


def test_the_face_id_array_can_be_named(measured):
    mesh = synthetic_mesh.cube_mesh(face_id_array_name="MaterialIds")
    assert len(faces.measure_faces(mesh, "MaterialIds")) == 3


def test_a_table_must_name_its_faces_uniquely():
    with pytest.raises(FaceTableError, match="unique"):
        FaceTable([Face(1, "wall"), Face(2, "cap_a"), Face(3, "cap_a")])


def test_a_table_must_classify_every_face():
    with pytest.raises(FaceTableError, match="cap_"):
        FaceTable([Face(1, "wall"), Face(2, "inlet")])


def test_a_table_needs_a_wall_and_a_cap():
    with pytest.raises(FaceTableError, match="no-slip"):
        FaceTable([Face(2, "cap_a"), Face(3, "cap_b")])
    with pytest.raises(FaceTableError, match="nowhere to enter"):
        FaceTable([Face(1, "wall")])


def test_a_split_wall_is_still_a_wall():
    table = FaceTable([Face(1, "wall_conduit"), Face(2, "wall_LPA"), Face(3, "cap_a")])
    assert [face.name for face in table.walls] == ["wall_conduit", "wall_LPA"]
    assert [face.name for face in table.caps] == ["cap_a"]
