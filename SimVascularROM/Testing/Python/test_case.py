"""A whole case on a Y, through `svromutils` and svZeroDSolver where they are available."""

import json
import os

import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy

pytest.importorskip("vmtk")
pytest.importorskip("svromutils")

from svromsetup import case, network, results, solver, testing
from svromsetup.boundary_conditions import Inflow, RCR, Resistance

SOLVER = solver.find_solver(os.environ.get("SVZERODSOLVER"))
needs_solver = pytest.mark.skipif(SOLVER is None,
                                  reason="svzerod is not installed, and SVZERODSOLVER names no solver")


@pytest.fixture(scope="module")
def y_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp("y")
    case.write_surfaces(testing.y_surface(), testing.NAMES, directory)
    centerlines = case.compute_centerlines(directory, testing.INLET_ID)
    return directory, centerlines


def end_of(centerlines, branch):
    branch_ids = vtk_to_numpy(centerlines.GetPointData().GetArray("BranchId"))
    points = vtk_to_numpy(centerlines.GetPoints().GetData())
    return points[np.flatnonzero(branch_ids == branch)[-1]]


def test_the_faces_are_written_as_mesh_prep_would(y_case):
    directory, _centerlines = y_case
    surface, faces = case.surface_paths(directory)
    assert surface.is_file()
    assert sorted(path.stem for path in faces.glob("*.vtp")) == sorted(testing.NAMES.values())


def test_each_outlet_is_named_after_the_cap_its_centerline_ends_at(y_case):
    """Asked geometrically, as everything that pairs a face with a vessel has to be: a list of
    names in the wrong order is the same names, all valid, and reads as right everywhere but in
    space."""
    directory, centerlines = y_case
    assert sorted(centerlines.outlet_names) == ["cap_left", "cap_right"]
    caps = testing.cap_centres()
    by_name = {testing.NAMES[face_id]: centre for face_id, centre in caps.items()}

    conditions = {"cap_inlet": Inflow.steady(10.0), "cap_right": Resistance(1000.0),
                  "cap_left": Resistance(3000.0)}
    config = json.loads(case.write_solver_input(
        directory, conditions, testing.INLET_ID, "cap_inlet",
        case.SimulationParameters(cardiac_cycles=2, points_per_cycle=20), "y").read_text())
    for vessel in config["vessels"]:
        outlet = vessel.get("boundary_conditions", {}).get("outlet")
        if not outlet:
            continue
        branch = int(vessel["vessel_name"].split("_")[0][len("branch"):])
        face = outlet[len("RESISTANCE_"):]
        nearest = min(by_name, key=lambda name: np.linalg.norm(by_name[name] - end_of(
            centerlines.geometry, branch)))
        assert face == nearest, f"{outlet} is on the vessel that ends at {nearest}"


def test_the_network_puts_each_outlet_after_the_vessel_that_reaches_its_cap(y_case):
    """Asked geometrically too: the graph names a condition after its cap, and the vessel it hangs
    off has to end at that cap in space, or the graph is a right-looking picture of the wrong
    model."""
    directory, centerlines = y_case
    config = case.write_solver_input(
        directory, {"cap_inlet": Inflow.steady(10.0), "cap_right": RCR(100.0, 1e-4, 1000.0, 0.0),
                    "cap_left": Resistance(3000.0)},
        testing.INLET_ID, "cap_inlet", case.SimulationParameters(cardiac_cycles=2, points_per_cycle=20), "y")
    net = network.network_from_config(config, "cap_inlet")
    assert net.nodes[net.root].label == "cap_inlet"
    caps = {testing.NAMES[face_id]: centre for face_id, centre in testing.cap_centres().items()}
    for cap in ("cap_left", "cap_right"):
        condition = net.node_for_cap(cap)
        (vessel,) = net.parents(condition.key)
        lines = results.vessel_lines(centerlines.geometry, config, net.nodes[vessel].vessel)
        points = vtk_to_numpy(lines.GetPoints().GetData())
        far = points[np.argmax(np.linalg.norm(points - caps["cap_inlet"], axis=1))]
        assert min(caps, key=lambda name: np.linalg.norm(caps[name] - far)) == cap


def test_the_solver_input_says_what_was_set_up(y_case):
    directory, _centerlines = y_case
    conditions = {"cap_inlet": Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0)),
                  "cap_right": RCR(100.0, 1e-4, 1000.0, 0.0), "cap_left": Inflow.steady(2.0)}
    parameters = case.SimulationParameters(cardiac_cycles=4, points_per_cycle=40)
    path = case.write_solver_input(directory, conditions, testing.INLET_ID, "cap_inlet",
                                   parameters, "y")
    config = json.loads(path.read_text())
    assert config["simulation_parameters"]["number_of_cardiac_cycles"] == 4
    assert config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] == 40
    written = {bc["bc_name"]: bc for bc in config["boundary_conditions"]}
    assert set(written) == {"INFLOW", "RCR_cap_right", "FLOW_cap_left"}
    # The steady inflow is spread over the inlet waveform's cycle, so the two agree on a period.
    assert written["FLOW_cap_left"]["bc_values"]["t"] == [0.0, 0.8]
    # Entering at an outlet end, which svZeroDSolver counts as negative.
    assert written["FLOW_cap_left"]["bc_values"]["Q"] == [-2.0, -2.0]
    # Lengths in cm, from a model in mm.
    lengths = [vessel["vessel_length"] for vessel in config["vessels"]]
    assert 0.5 < sum(lengths) < 10.0


def test_a_cap_left_without_a_condition_is_refused_by_name(y_case):
    directory, _centerlines = y_case
    with pytest.raises(case.CaseError, match="cap_left"):
        case.write_solver_input(directory, {"cap_inlet": Inflow.steady(1.0),
                                            "cap_right": Resistance(1.0)},
                                testing.INLET_ID, "cap_inlet", case.SimulationParameters(), "y")


@needs_solver
def test_flow_is_conserved_through_the_model(y_case):
    directory, centerlines = y_case
    conditions = {"cap_inlet": Inflow.steady(10.0), "cap_right": Resistance(2000.0),
                  "cap_left": Inflow.steady(3.0)}
    config = case.write_solver_input(directory, conditions, testing.INLET_ID, "cap_inlet",
                                     case.SimulationParameters(cardiac_cycles=3,
                                                               points_per_cycle=20), "y")
    solver.run_solver(config, directory / case.RESULTS_NAME, SOLVER)
    solved = solver.read_results(directory / case.RESULTS_NAME)
    faces = {face.name: face for face in results.face_results(config, solved, "cap_inlet")}
    assert faces["cap_right"].mean_flow == pytest.approx(13.0, rel=1e-6)
    assert faces["cap_right"].mean_pressure_mmhg == pytest.approx(13.0 * 2000.0 / 1333.22, rel=1e-3)

    coloured = results.centerline_results(centerlines.geometry, config, solved)
    pressure = vtk_to_numpy(coloured.GetPointData().GetArray(results.PRESSURE_ARRAY_NAME))
    assert pressure.min() == pytest.approx(faces["cap_right"].mean_pressure_mmhg, rel=1e-3)
    assert pressure.max() <= faces["cap_left"].mean_pressure_mmhg + 1e-6
