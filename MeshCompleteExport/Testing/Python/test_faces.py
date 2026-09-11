import pytest

from svmeshcomplete import testing as synthetic_mesh
from svmeshcomplete import faces
from svmeshcomplete.face_table import (
    Face,
    FaceTable,
    FaceTableError,
    read_clip_vessel_table,
    starter_rows,
)

CLIP_VESSEL_CSV = '''"LabelValue","Name","Color_R","Color_G","Color_B"
1,"Wall",199,199,207
2,"Inlet",180,242,92
3,"Outlet 1",242,92,224
15,"Outlet 13",149,242,92
'''


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


def test_clip_vessel_names_are_slugged_and_numbers_padded():
    labels = read_clip_vessel_table_from(CLIP_VESSEL_CSV)
    names = {int(row["FaceID"]): row["Name"] for row in starter_rows(labels)}
    assert names == {1: "wall", 2: "cap_inlet", 3: "cap_outlet_01", 15: "cap_outlet_13"}


def read_clip_vessel_table_from(text):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "Clip Vessel face colors.csv"
    path.write_text(text)
    return read_clip_vessel_table(path)


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


def test_a_table_round_trips_through_its_file(tmp_path):
    table = FaceTable([Face(1, "wall", "Wall"), Face(2, "cap_RSVC", "Inlet")])
    table.write(tmp_path / "face_table.csv")
    reread = FaceTable.read(tmp_path / "face_table.csv")
    assert reread.names_by_id() == {1: "wall", 2: "cap_RSVC"}
    assert reread.faces[1].clip_vessel_name == "Inlet"


def test_a_non_colour_table_is_not_read_as_one(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(FaceTableError, match="LabelValue"):
        read_clip_vessel_table(path)
