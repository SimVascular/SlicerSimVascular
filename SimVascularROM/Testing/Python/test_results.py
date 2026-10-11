"""Reading svZeroDSolver's results back per face, without a centerline or the package."""

import json
import os

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svromsetup import results, solver
from svromsetup.boundary_conditions import MMHG


def vessel(index, bc=None, length=1.0):
    entry = {"vessel_id": index, "vessel_name": f"branch{index}_seg0", "vessel_length": length,
             "zero_d_element_type": "BloodVessel",
             "zero_d_element_values": {"R_poiseuille": 10.0, "C": 1e-5, "L": 1.0}}
    if bc:
        entry["boundary_conditions"] = bc
    return entry


# A Y with an inflow at its inlet, a resistance on one outlet, and a second inflow prescribed on
# the other outlet's end -- which is how the package writes every inflow but the inlet's.
CONFIG = {
    "simulation_parameters": {"number_of_cardiac_cycles": 3,
                              "number_of_time_pts_per_cardiac_cycle": 11,
                              "density": 1.06, "viscosity": 0.04},
    "vessels": [vessel(0, {"inlet": "INFLOW"}), vessel(1, {"outlet": "RESISTANCE_cap_right"}),
                vessel(2, {"outlet": "FLOW_cap_left"})],
    "junctions": [{"junction_name": "J0", "junction_type": "NORMAL_JUNCTION",
                   "inlet_vessels": [0], "outlet_vessels": [1, 2]}],
    "boundary_conditions": [
        {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"t": [0.0, 1.0], "Q": [10.0, 10.0]}},
        {"bc_name": "RESISTANCE_cap_right", "bc_type": "RESISTANCE",
         "bc_values": {"R": 100.0, "Pd": 0.0}},
        {"bc_name": "FLOW_cap_left", "bc_type": "FLOW",
         "bc_values": {"t": [0.0, 1.0], "Q": [-5.0, -5.0]}},
    ],
}


def write_results(path, rows):
    with open(path, "w") as handle:
        handle.write(",".join(solver.RESULT_COLUMNS) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return path


@pytest.fixture
def solved(tmp_path):
    """What svZeroDSolver writes for CONFIG, steady: 10 in, 5 more in at the left, 15 out."""
    rows = []
    for time in np.linspace(0.0, 1.0, 11):
        rows += [("branch0_seg0", time, 10.0, 10.0, 1750.0, 1650.0),
                 ("branch1_seg0", time, 15.0, 15.0, 1650.0, 1500.0),
                 ("branch2_seg0", time, -5.0, -5.0, 1650.0, 1700.0)]
    return solver.read_results(write_results(tmp_path / "results.csv", rows))


def test_the_results_are_read_per_vessel(solved):
    assert set(solved) == {"branch0_seg0", "branch1_seg0", "branch2_seg0"}
    assert solved["branch1_seg0"].flow_out == pytest.approx(np.full(11, 15.0))


def test_a_file_that_is_not_the_solvers_is_refused(tmp_path):
    (tmp_path / "other.csv").write_text("a,b,c\n1,2,3\n")
    with pytest.raises(solver.SolverError, match="columns"):
        solver.read_results(tmp_path / "other.csv")


def test_every_cap_reads_positive_in_the_direction_its_condition_drives_it(solved):
    """The inflow prescribed at an outlet end is negative in the solver's own terms; turned round
    here, so that every inflow reads as flow in and every outlet as flow out."""
    faces = {face.name: face for face in results.face_results(CONFIG, solved, "cap_inlet")}
    assert set(faces) == {"cap_inlet", "cap_right", "cap_left"}
    assert faces["cap_inlet"].mean_flow == pytest.approx(10.0)
    assert faces["cap_left"].mean_flow == pytest.approx(5.0)
    assert faces["cap_left"].is_inflow and not faces["cap_right"].is_inflow
    assert faces["cap_right"].mean_flow == pytest.approx(15.0)
    # Pressure at the cap's own end of its vessel, in mmHg.
    assert faces["cap_inlet"].mean_pressure_mmhg == pytest.approx(1750.0 / MMHG)
    assert faces["cap_right"].mean_pressure_mmhg == pytest.approx(1500.0 / MMHG)
    assert faces["cap_left"].end == "outlet"


def test_the_exported_csv_has_a_flow_and_a_pressure_column_per_cap(solved, tmp_path):
    faces = results.face_results(CONFIG, solved, "cap_inlet")
    path = results.write_face_results_csv(tmp_path / "out" / "caps.csv", faces)
    header, *rows = path.read_text().splitlines()
    assert header.split(",") == [
        "time [s]",
        "cap_inlet flow [mL/s]", "cap_inlet pressure [mmHg]",
        "cap_right flow [mL/s]", "cap_right pressure [mmHg]",
        "cap_left flow [mL/s]", "cap_left pressure [mmHg]",
    ]
    assert len(rows) == 11
    first = [float(value) for value in rows[0].split(",")]
    # The inflow at an outlet end reads positive, as it does in the panel, and pressure in mmHg.
    assert first[5] == pytest.approx(5.0)
    assert first[4] == pytest.approx(1500.0 / MMHG)
    with pytest.raises(ValueError, match="no results"):
        results.write_face_results_csv(tmp_path / "empty.csv", [])


def test_the_centerlines_carry_the_pressure_along_each_branch(solved):
    """Three straight branches meeting at a bifurcation point, as the package splits them."""
    points = vtk.vtkPoints()
    branch_ids, paths = [], []
    for branch, direction in ((0, (0, 0, -1)), (1, (1, 0, 1)), (2, (-1, 0, 1))):
        for step in range(5):
            points.InsertNextPoint(*(np.array(direction, float) * (step + 1)))
            branch_ids.append(branch)
            paths.append(float(step))
    points.InsertNextPoint(0.0, 0.0, 0.0)
    branch_ids.append(-1)
    paths.append(0.0)
    centerline = vtk.vtkPolyData()
    centerline.SetPoints(points)
    for name, values in (("BranchId", np.array(branch_ids, dtype=np.int32)),
                         ("Path", np.array(paths))):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        centerline.GetPointData().AddArray(array)

    coloured = results.centerline_results(centerline, CONFIG, solved)
    pressure = vtk_to_numpy(coloured.GetPointData().GetArray(results.PRESSURE_ARRAY_NAME))
    flow = vtk_to_numpy(coloured.GetPointData().GetArray(results.FLOW_ARRAY_NAME))
    # Branch 1 runs from 1650 at its start to 1500 at its end, linearly along its path.
    assert pressure[5:10] == pytest.approx(np.linspace(1650.0, 1500.0, 5) / MMHG)
    assert flow[5:10] == pytest.approx(np.full(5, 15.0))
    # The bifurcation point takes its nearest branch point's value rather than none.
    assert np.isfinite(pressure[-1]) and pressure[-1] > 0
    assert centerline.GetPointData().GetArray(results.PRESSURE_ARRAY_NAME) is None, \
        "the centerlines passed in are left as they were"


SOLVER = solver.find_solver(os.environ.get("SVZERODSOLVER"))


@pytest.mark.skipif(SOLVER is None, reason="svzerod is not installed, and SVZERODSOLVER names no solver")
def test_a_flow_prescribed_on_an_outlet_end_enters_the_model(tmp_path):
    """The sign the package writes a second inflow with, held to what the solver does with it."""
    config = tmp_path / "solver_0d.json"
    config.write_text(json.dumps(CONFIG))
    solver.run_solver(config, tmp_path / "results.csv", SOLVER)
    faces = {face.name: face for face in results.face_results(
        CONFIG, solver.read_results(tmp_path / "results.csv"), "cap_inlet")}
    assert faces["cap_left"].mean_flow == pytest.approx(5.0)
    assert faces["cap_right"].mean_flow == pytest.approx(15.0)
    assert faces["cap_right"].mean_pressure_mmhg == pytest.approx(15.0 * 100.0 / MMHG)


@pytest.mark.skipif(SOLVER is None, reason="svzerod is not installed, and SVZERODSOLVER names no solver")
def test_a_failed_solve_says_so_and_leaves_no_results_behind(tmp_path):
    broken = dict(CONFIG, boundary_conditions=CONFIG["boundary_conditions"][:1])
    config = tmp_path / "solver_0d.json"
    config.write_text(json.dumps(broken))
    (tmp_path / "results.csv").write_text("left over from an earlier run")
    with pytest.raises(solver.SolverError, match="failed"):
        solver.run_solver(config, tmp_path / "results.csv", SOLVER)
    assert not (tmp_path / "results.csv").exists()
