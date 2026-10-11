"""A rigid-wall case for svmeshcomplete's cube: wall 1, cap_inlet 2 (bottom), cap_outlet 3 (top)."""

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from svmeshcomplete import testing
from svmpsetup import case, settings
from svromsetup.boundary_conditions import RCR, Inflow, Resistance

NAMES = {testing.WALL_ID: "wall", testing.INLET_ID: "cap_inlet", testing.OUTLET_ID: "cap_outlet"}
WAVEFORM = Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0), "/somewhere/cap_inlet.flow")


def written(tmp_path, conditions, values=None):
    result = case.write_case(tmp_path, testing.cube_mesh(), NAMES, conditions, values)
    return result, ET.parse(result.solver_xml).getroot()


def bc(root, name):
    return root.find(f"Add_equation[@type='fluid']/Add_BC[@name='{name}']")


def test_every_face_of_the_mesh_is_in_the_file_and_the_wall_does_not_slip(tmp_path):
    """A face no condition names is solved as zero traction: on a wall, a hole."""
    result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                      "cap_outlet": Resistance(1000.0)})
    faces = {face.get("name"): face.findtext("Face_file_path")
             for face in root.findall("Add_mesh/Add_face")}
    assert faces == {"wall": "mesh/mesh-surfaces/wall.vtp",
                     "cap_inlet": "mesh/mesh-surfaces/cap_inlet.vtp",
                     "cap_outlet": "mesh/mesh-surfaces/cap_outlet.vtp"}
    wall = bc(root, "wall")
    assert (wall.findtext("Type"), wall.findtext("Time_dependence"), wall.findtext("Value")) \
        == ("Dir", "Steady", "0.0")
    assert root.findtext("Add_mesh/Mesh_file_path") == "mesh/mesh-complete.mesh.vtu"
    assert (tmp_path / "mesh" / "mesh-complete.mesh.vtu").is_file()
    assert result.faces == 3 and result.caps == 2


def test_the_scale_factor_is_always_written(tmp_path):
    """Left out, the solver takes 1.0, and a mesh in mm becomes a domain ten times too large."""
    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                       "cap_outlet": Resistance(1000.0)})
    assert root.findtext("Add_mesh/Mesh_scale_factor") == "0.1"


def test_an_inflow_enters_the_domain_negative_and_a_waveform_gets_its_own_file(tmp_path):
    result, root = written(tmp_path, {"cap_inlet": WAVEFORM, "cap_outlet": Resistance(1000.0)})
    inlet = bc(root, "cap_inlet")
    assert inlet.findtext("Type") == "Dir"
    assert inlet.findtext("Time_dependence") == "Unsteady"
    assert inlet.findtext("Temporal_values_file_path") == "cap_inlet.flow"
    assert inlet.findtext("Profile") == "Parabolic"
    assert inlet.findtext("Impose_flux") == "true"
    lines = (tmp_path / "cap_inlet.flow").read_text().splitlines()
    # Points, then Fourier coefficients -- not columns -- then the series, flow negated.
    assert lines[0] == "3 16"
    assert lines[1:] == ["0.0 -5.0", "0.4 -15.0", "0.8 -5.0"]
    assert result.flow_files == ("cap_inlet.flow",)

    # A steady flow typed rather than loaded is a value, negative, and needs no file.
    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(12.5),
                                       "cap_outlet": Resistance(1000.0)})
    inlet = bc(root, "cap_inlet")
    assert (inlet.findtext("Time_dependence"), inlet.findtext("Value")) == ("Steady", "-12.5")
    assert not (tmp_path / "cap_inlet.flow").exists(), "the earlier waveform is not left behind"


def test_an_rcr_starts_at_its_initial_pressure_and_a_resistance_has_no_distal_pressure(tmp_path):
    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                       "cap_outlet": RCR(121.0, 1.5e-4, 1212.0, 0.0)})
    rcr = bc(root, "cap_outlet")
    assert (rcr.findtext("Type"), rcr.findtext("Time_dependence")) == ("Neu", "RCR")
    values = {child.tag: child.text for child in rcr.find("RCR_values")}
    assert values == {"Capacitance": "0.00015", "Distal_resistance": "1212.0",
                      "Proximal_resistance": "121.0", "Distal_pressure": "0.0",
                      "Initial_pressure": "0.0"}

    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                       "cap_outlet": Resistance(1000.0)})
    resistance = bc(root, "cap_outlet")
    assert [child.tag for child in resistance] == ["Type", "Time_dependence", "Value"]
    assert resistance.findtext("Time_dependence") == "Resistance"


def test_the_linear_solver_is_the_rigid_wall_default_exactly(tmp_path):
    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                       "cap_outlet": Resistance(1000.0)})
    ls = root.find("Add_equation[@type='fluid']/LS")
    assert ls.get("type") == "NS"
    assert ls.find("Linear_algebra").get("type") == "fsils"
    assert ls.findtext("Linear_algebra/Preconditioner") == "fsils"
    assert {child.tag: child.text for child in ls if child.tag != "Linear_algebra"} == {
        "Max_iterations": "10", "NS_GM_max_iterations": "200", "NS_CG_max_iterations": "500",
        "Tolerance": "0.4", "NS_GM_tolerance": "0.01", "NS_CG_tolerance": "0.2",
        "Krylov_space_dimension": "50"}


def test_a_changed_setting_is_written_and_a_bad_one_refused_by_name(tmp_path):
    _result, root = written(tmp_path, {"cap_inlet": Inflow.steady(10.0),
                                       "cap_outlet": Resistance(1000.0)},
                            {"general.number_of_time_steps": "40", "fluid.viscosity": 0.035,
                             "output.vorticity": False})
    assert root.findtext("GeneralSimulationParameters/Number_of_time_steps") == "40"
    assert root.findtext("Add_equation[@type='fluid']/Viscosity/Value") == "0.035"
    spatial = root.find("Add_equation[@type='fluid']/Output[@type='Spatial']")
    assert spatial.find("Vorticity") is None and spatial.findtext("Velocity") == "true"

    # What svMultiPhysics would read as its longest numeric prefix, without a word.
    with pytest.raises(case.CaseError, match="Time step size"):
        written(tmp_path, {"cap_inlet": Inflow.steady(10.0), "cap_outlet": Resistance(1.0)},
                {"general.time_step_size": "1,5e-4"})
    with pytest.raises(settings.SettingError, match="No such setting"):
        settings.resolve({"general.time_step": 1e-3})


def test_a_case_that_cannot_close_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(case.CaseError, match="cap_outlet"):
        written(tmp_path, {"cap_inlet": Inflow.steady(10.0)})
    with pytest.raises(case.CaseError, match="nothing sets the pressure"):
        written(tmp_path, {"cap_inlet": Inflow.steady(10.0), "cap_outlet": Inflow.steady(1.0)})
    assert not (tmp_path / "solver.xml").exists()


def test_a_surface_is_not_a_volume_mesh(tmp_path):
    from svmeshcomplete import faces
    surface = faces.boundary_of(testing.cube_mesh())
    with pytest.raises(case.CaseError, match="volume mesh"):
        case.write_case(tmp_path, surface, NAMES, {"cap_inlet": Inflow.steady(1.0),
                                                   "cap_outlet": Resistance(1.0)})


def test_the_check_finds_what_the_solver_would_only_find_on_the_cluster(tmp_path):
    result, _root = written(tmp_path, {"cap_inlet": WAVEFORM, "cap_outlet": Resistance(1000.0)})
    (tmp_path / "cap_inlet.flow").unlink()
    (tmp_path / "mesh" / "mesh-surfaces" / "cap_stray.vtp").write_text("")
    with pytest.raises(case.CaseError) as raised:
        case.check_case(result.solver_xml)
    assert "cap_inlet.flow" in str(raised.value)
    assert "cap_stray" in str(raised.value)


SOLVER = os.environ.get("SVMULTIPHYSICS") or shutil.which("svmultiphysics")


@pytest.mark.skipif(not SOLVER, reason="svmultiphysics is not on the PATH or at SVMULTIPHYSICS")
def test_svmultiphysics_reads_the_case_and_steps_it(tmp_path):
    """The one judge of whether a solver.xml is right is the solver: two steps, and results."""
    values = {"general.number_of_time_steps": 2, "general.time_step_size": 1e-3,
              "general.start_saving_after_time_step": 1, "general.increment_in_saving_vtk_files": 1,
              "general.increment_in_saving_restart_files": 1}
    result, _root = written(tmp_path, {"cap_inlet": WAVEFORM,
                                       "cap_outlet": RCR(121.0, 1.5e-4, 1212.0, 0.0)}, values)
    completed = subprocess.run([SOLVER, result.solver_xml.name], cwd=tmp_path,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert list(tmp_path.glob("*-procs/result_*.vtu")), "no results were written"
