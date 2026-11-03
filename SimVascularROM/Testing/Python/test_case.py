"""A whole case on svReducedOrderModelingUtils' testing aorta, through `svromutils` and svZeroDSolver
where they are available.

The aorta is `case01` of the package's TESTING_DATA release, which `svromsetup.testing_data`
fetches and reads: a real model rather than a made one, so that what is checked is what a user's
case goes through -- faces named by their files, conditions read from SimVascular's, a waveform
of 256 points -- and the centerlines can be held to ones that came from somewhere else.

The files are downloaded once, into pytest's cache, and the tests are skipped where they cannot
be: offline, set `SVROM_TESTING_DATA` to a folder holding them as the release has them.
"""

import json
import os

import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy

pytest.importorskip("vmtk")
pytest.importorskip("svromutils")

from svmeshcomplete import io
from svromsetup import boundary_conditions as bcs
from svromsetup import case, network, results, solver, testing_data
from svromsetup.boundary_conditions import RCR, Resistance
from svromsetup.testing_data import INLET, RCR_OUTLET

SOLVER = solver.find_solver(os.environ.get("SVZERODSOLVER"))
needs_solver = pytest.mark.skipif(SOLVER is None,
                                  reason="svzerod is not installed, and SVZERODSOLVER names no solver")

# Enough cycles for the RCR's compliance to settle -- Rd times C is 1.3 s, a cycle 0.91 s -- so
# that a mean over the last one is the periodic mean, and what goes in has come out.
PARAMETERS = case.SimulationParameters(cardiac_cycles=10, points_per_cycle=256)


@pytest.fixture(scope="module")
def data(request, tmp_path_factory):
    """The aorta: downloaded into pytest's cache, kept between runs -- or, with the cache turned off,
    into this run's own folder -- unless `SVROM_TESTING_DATA` names one that holds it."""
    cache = getattr(request.config, "cache", None)
    downloads = (cache.mkdir("svromutils-testing-data") if cache is not None
                 else tmp_path_factory.mktemp("svromutils-testing-data"))
    try:
        return testing_data.case01(downloads, tmp_path_factory.mktemp("case01"))
    except testing_data.TestingDataUnavailable as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def faces(data):
    """`{face id: name}` and `{name: centre}`, from the mesh-surfaces files as SimVascular names them."""
    return data.names, data.centres


@pytest.fixture(scope="module")
def conditions(data):
    return data.conditions


@pytest.fixture(scope="module")
def aorta(data, tmp_path_factory):
    """The case folder, its faces written and its centerlines computed, as the panel does both."""
    directory = tmp_path_factory.mktemp("aorta")
    case.write_surfaces(data.surface(), data.names, directory, testing_data.FACE_ID_ARRAY)
    inlet_id = data.face_id(INLET)
    return directory, inlet_id, case.compute_centerlines(directory, inlet_id)


def solver_input(aorta, conditions, parameters=PARAMETERS):
    directory, inlet_id, _centerlines = aorta
    return case.write_solver_input(directory, conditions, inlet_id, INLET, parameters, "case01")


def distances(points, to):
    """Each point's distance to the nearest of `to`."""
    squared = (points ** 2).sum(axis=1)[:, None] + (to ** 2).sum(axis=1)[None, :] - 2.0 * points @ to.T
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def test_the_files_give_every_cap_but_the_inlet_its_condition(faces, conditions):
    """Read as SimVascular wrote them, by the names the faces' files go by -- and the inflow from its
    first row, `0 82.03541779`, whose time is written without a decimal point."""
    names, _centres = faces
    caps = sorted(name for name in names.values() if name.startswith("cap_"))
    assert sorted(conditions) == caps
    assert conditions[RCR_OUTLET] == RCR(180.0, 0.0008, 1620.0, 0.0)
    assert all(conditions[name] == Resistance(16000.0) for name in caps
               if name not in (INLET, RCR_OUTLET))
    inflow = conditions[INLET]
    assert (len(inflow.time), inflow.time[0], inflow.flow[0]) == (256, 0.0, 82.03541779)
    assert bcs.as_inflow(inflow) == (inflow, False), "written positive, into the model"


def test_the_faces_are_written_as_mesh_prep_would(data, faces, aorta):
    """One file per face, named, each the face SimVascular's file of that name is."""
    mesh = data.mesh_dir
    names, _centres = faces
    directory, _inlet_id, _centerlines = aorta
    surface, written = case.surface_paths(directory)
    assert surface.is_file()
    assert sorted(path.stem for path in written.glob("*.vtp")) == sorted(names.values())
    for name in names.values():
        assert (io.read_dataset(written / f"{name}.vtp").GetNumberOfCells()
                == io.read_dataset(mesh / "mesh-surfaces" / f"{name}.vtp").GetNumberOfCells()), name


def test_the_centerlines_are_the_releases_own(data, faces, aorta):
    """Held to the package's own centerlines of the same aorta, which came from elsewhere: every
    point of either within 2 mm of the other, and a tenth of a millimetre apart on average, in an
    aorta some 25 mm across. Splitting into branches is not compared -- the release splits one
    branch more -- only where the lines are."""
    names, _centres = faces
    _directory, _inlet_id, centerlines = aorta
    assert sorted(centerlines.outlet_names) == sorted(
        name for name in names.values() if name.startswith("cap_") and name != INLET)
    ours = vtk_to_numpy(centerlines.geometry.GetPoints().GetData())
    theirs = vtk_to_numpy(io.read_dataset(data.centerlines_path).GetPoints().GetData())
    for there, back in ((ours, theirs), (theirs, ours)):
        apart = distances(there, back)
        assert apart.max() < 2.0 and apart.mean() < 0.5, (apart.max(), apart.mean())


def test_each_outlet_is_named_after_the_cap_its_centerline_ends_at(faces, conditions, aorta):
    """Asked geometrically, as everything that pairs a face with a vessel has to be: a list of
    names in the wrong order is the same names, all valid, and reads as right everywhere but in
    space -- and here four of the five outlets are resistances of 16000, alike in every number."""
    _names, centres = faces
    _directory, _inlet_id, centerlines = aorta
    config = json.loads(solver_input(aorta, conditions).read_text())
    branch_ids = vtk_to_numpy(centerlines.geometry.GetPointData().GetArray("BranchId"))
    points = vtk_to_numpy(centerlines.geometry.GetPoints().GetData())
    named = 0
    for vessel in config["vessels"]:
        outlet = vessel.get("boundary_conditions", {}).get("outlet")
        if not outlet:
            continue
        branch = int(vessel["vessel_name"].split("_")[0][len("branch"):])
        end = points[np.flatnonzero(branch_ids == branch)[-1]]
        face = outlet.split("_", 1)[1]
        nearest = min(centres, key=lambda name: np.linalg.norm(centres[name] - end))
        assert face == nearest, f"{outlet} is on the vessel that ends at {nearest}"
        named += 1
    assert named == 5


def test_the_network_puts_each_outlet_after_the_vessel_that_reaches_its_cap(faces, conditions, aorta):
    """Asked geometrically too: the graph names a condition after its cap, and the vessel it hangs
    off has to end at that cap in space, or the graph is a right-looking picture of the wrong
    model."""
    _names, centres = faces
    _directory, _inlet_id, centerlines = aorta
    config = solver_input(aorta, conditions)
    net = network.network_from_config(config, INLET)
    assert net.nodes[net.root].label == INLET
    for cap in (name for name in conditions if name != INLET):
        (vessel,) = net.parents(net.node_for_cap(cap).key)
        lines = results.vessel_lines(centerlines.geometry, config, net.nodes[vessel].vessel)
        points = vtk_to_numpy(lines.GetPoints().GetData())
        far = points[np.argmax(np.linalg.norm(points - centres[INLET], axis=1))]
        assert min(centres, key=lambda name: np.linalg.norm(centres[name] - far)) == cap


def test_the_solver_input_says_what_was_set_up(data, faces, conditions, aorta):
    """Every file's values on the condition of its name, the run asked for, and lengths in cm from
    a model in mm: the vessels from the valve to each cap add up to between half and all of the
    release's centerline to that cap -- not all, since a junction between branches is no vessel,
    and a model taken for centimetres would be ten times off either way."""
    _names, centres = faces
    config = json.loads(solver_input(aorta, conditions).read_text())
    assert config["simulation_parameters"]["number_of_cardiac_cycles"] == PARAMETERS.cardiac_cycles
    assert config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] == PARAMETERS.points_per_cycle
    written = {bc["bc_name"]: bc["bc_values"] for bc in config["boundary_conditions"]}
    assert set(written) == {"INFLOW", f"RCR_{RCR_OUTLET}"} | {
        f"RESISTANCE_{name}" for name, condition in conditions.items() if isinstance(condition, Resistance)}
    assert written["INFLOW"]["Q"] == pytest.approx(conditions[INLET].flow, rel=1e-9)
    assert written["INFLOW"]["t"] == pytest.approx(conditions[INLET].time, rel=1e-9)
    rcr = written[f"RCR_{RCR_OUTLET}"]
    assert (rcr["Rp"], rcr["C"], rcr["Rd"], rcr["Pd"]) == (180.0, 0.0008, 1620.0, 0.0)
    assert all(written[f"RESISTANCE_{name}"]["R"] == 16000.0 for name, condition in conditions.items()
               if isinstance(condition, Resistance))

    theirs = testing_data.release_centerlines(data.centerlines_path, centres[INLET])
    lengths = {vessel["vessel_name"]: vessel["vessel_length"] for vessel in config["vessels"]}
    net = network.network_from_config(config, INLET)
    for cap in (name for name in conditions if name != INLET):
        key, centimetres = net.node_for_cap(cap).key, 0.0
        while net.parents(key):
            (key,) = net.parents(key)
            centimetres += lengths.get(net.nodes[key].vessel, 0.0)
        _end, millimetres = min(theirs, key=lambda line: np.linalg.norm(line[0] - centres[cap]))
        assert 0.5 < centimetres * 10.0 / millimetres < 1.0, (cap, centimetres, millimetres)


def test_a_cap_left_without_a_condition_is_refused_by_name(conditions, aorta):
    left_out = {name: condition for name, condition in conditions.items() if name != "cap_left_subclavian"}
    with pytest.raises(case.CaseError, match="cap_left_subclavian"):
        solver_input(aorta, left_out, case.SimulationParameters(cardiac_cycles=2, points_per_cycle=20))


@needs_solver
def test_every_outlet_drains_as_its_file_says_and_what_goes_in_comes_out(conditions, aorta):
    """Over the last cycle: the valve's mean flow the file's; each resistance's flow its pressure
    over its R, as it is at every instant; the RCR's its pressure less Pd over Rp plus Rd, as it is
    on average once the cycle repeats; the outflows adding up to the inflow; and the centerlines
    coloured from the valve's pressure down to the lowest outlet's."""
    directory, _inlet_id, centerlines = aorta
    config = solver_input(aorta, conditions)
    solver.run_solver(config, directory / case.RESULTS_NAME, SOLVER)
    solved = solver.read_results(directory / case.RESULTS_NAME)
    faces = {face.name: face for face in results.face_results(config, solved, INLET)}
    assert sorted(faces) == sorted(conditions)
    inflow = faces[INLET].mean_flow
    assert inflow == pytest.approx(conditions[INLET].mean, rel=1e-6)
    for name, condition in conditions.items():
        pressure = faces[name].mean_pressure_mmhg * 1333.22
        if isinstance(condition, Resistance):
            assert faces[name].mean_flow == pytest.approx(pressure / condition.R, rel=1e-4), name
        elif isinstance(condition, RCR):
            assert faces[name].mean_flow == pytest.approx(
                (pressure - condition.Pd) / (condition.Rp + condition.Rd), rel=1e-3), name
    assert sum(face.mean_flow for name, face in faces.items() if name != INLET) == pytest.approx(inflow, rel=1e-5)

    coloured = results.centerline_results(centerlines.geometry, config, solved)
    pressure = vtk_to_numpy(coloured.GetPointData().GetArray(results.PRESSURE_ARRAY_NAME))
    outlets = [face.mean_pressure_mmhg for name, face in faces.items() if name != INLET]
    assert pressure.max() == pytest.approx(faces[INLET].mean_pressure_mmhg, rel=1e-3)
    assert pressure.min() == pytest.approx(min(outlets), rel=1e-3)
