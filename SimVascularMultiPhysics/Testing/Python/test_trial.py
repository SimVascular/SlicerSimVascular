"""A few steps of a case before the full run: finding the solver, the test's file, and reading
what the solver prints -- on lines svMultiPhysics printed, and on the cube when it is here."""

import os
import shutil
import stat
import subprocess
import xml.etree.ElementTree as ET

import pytest

from svmeshcomplete import testing
from svmpsetup import case, trial
from svromsetup.boundary_conditions import RCR, Inflow, Resistance

NAMES = {testing.WALL_ID: "wall", testing.INLET_ID: "cap_inlet", testing.OUTLET_ID: "cap_outlet"}
CONDITIONS = {"cap_inlet": Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0), "/somewhere/cap_inlet.flow"),
              "cap_outlet": RCR(121.0, 1.5e-4, 1212.0, 0.0)}

# The thoracic aorta of the tutorial, two steps on 8 processes: step 1 stopped at the iteration
# limit with its residual at 2e-2 of its first, step 2 just under the 1e-3 tolerance.
AORTA = """\
 Eq     N-i     T       dB  Ri/R1   Ri/R0    R/Ri     lsIt   dB  %t
 NS 1-1  9.465e+00  [0 1.000e+00 1.000e+00 3.016e-01]  [1 -8 84]
 NS 1-4  2.293e+01  [-22 7.824e-02 7.824e-02 2.490e-01]  [0 -14 28]
 NS 1-5s 2.527e+01  [-33 2.043e-02 2.043e-02 3.566e-01]  [0 -10 32]
 NS 2-1  3.596e+01  [0 1.000e+00 3.592e+01 5.294e-02]  [0 -29 36]
 NS 2-5s 5.818e+01  [-60 9.050e-04 3.251e-02 2.058e-01]  [0 -16 34]
"""
LIMITS = trial.Schedule(steps=10000, save_increment=100, start_saving=1,
                        min_iterations=2, max_iterations=5, tolerance=1e-3)


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_an_iteration_line_says_its_step_and_iteration_and_s_is_a_save_not_convergence():
    line = trial.parse_iteration(" NS 1-5s 2.527e+01  [-33 2.043e-02 2.043e-02 3.566e-01]  [0 -10 32]")
    assert (line.step, line.iteration, line.saved, line.time) == (1, 5, True, 25.27)
    assert (line.ri_r1, line.ri_r0) == (2.043e-02, 2.043e-02)
    assert not line.residual_grew and not line.linear_failed
    marked = trial.parse_iteration(" NS 3-2  5.376e-02  !21 2.151e+01 4.178e-02 2.474e-01!  !0 -14 0!")
    assert marked.residual_grew and marked.linear_failed and not marked.saved
    assert trial.parse_iteration(" Eq     N-i     T       dB  Ri/R1   Ri/R0    R/Ri     lsIt   dB  %t") is None


def test_a_step_at_the_iteration_limit_above_the_tolerance_is_said_not_to_have_converged():
    progress = trial.Progress(2, LIMITS)
    for line in AORTA.splitlines():
        progress.feed(line)
    assert progress.step_seconds() == pytest.approx([25.27, 32.91])
    assert progress.seconds_per_step() == pytest.approx(32.91), "the first step pays for setting up"
    assert progress.unconverged() == [(1, 5, pytest.approx(2.043e-02))]
    report = progress.finish(0)
    assert report.ok and report.steps_done == 2
    assert any("Step 1 used all 5 nonlinear iterations" in note for note in report.notes)


def test_the_summary_times_the_full_run_and_its_results():
    progress = trial.Progress(2, LIMITS)
    for line in AORTA.splitlines():
        progress.feed(line)
    lines = trial.summary(progress.finish(0), LIMITS, 8, result_bytes=41.8e6)
    assert lines[0] == "2 of 2 steps, 33 s per step on 8 processors."
    assert "the full run, 10000 steps, takes about 3.8 days here" in lines[1]
    assert "100 result files come to about 4.2 GB" in lines[1]
    assert "No errors." in lines


def test_the_solvers_own_error_is_reported_with_its_reason_and_not_mpis_abort_after_it():
    progress = trial.Progress(5, LIMITS)
    for line in ("[svMultiPhysics] ERROR: The svMultiPhysics program has failed due to unhandled exception.",
                 "[construct_fluid] Jacobian for element 12 is < 0.",
                 "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD with errorcode 1"):
        progress.feed(line)
    report = progress.finish(1)
    assert report.errors == ["svMultiPhysics stopped with an error: [construct_fluid] Jacobian for "
                             "element 12 is < 0."]
    assert not report.ok


def test_what_mpi_and_the_loader_say_is_put_in_words():
    progress = trial.Progress(5, LIMITS)
    progress.feed("There are not enough slots available in the system to satisfy the 8")
    progress.feed("svmultiphysics: error while loading shared libraries: libmpi.so.40: cannot open "
                  "shared object file: No such file or directory")
    errors = progress.finish(1).errors
    assert errors[0].startswith("This machine has fewer cores than the processors asked for")
    assert "missing (libmpi.so.40)" in errors[1] and "apt-get install" in errors[1]


def test_a_nan_is_an_error_though_the_solver_carries_on():
    progress = trial.Progress(3, LIMITS)
    progress.feed(" NS 1-5s 2.527e+01  [-33 2.043e-02 2.043e-02 3.566e-01]  [0 -10 32]")
    progress.feed(" NS 2-1  3.596e+01  [0 nan nan nan]  !0 -29 36!")
    assert progress.nan_step == 2
    report = progress.finish(0, stopped=True)
    assert any("NaN at step 2" in error for error in report.errors)


def test_a_test_that_ends_early_without_a_word_is_an_error():
    progress = trial.Progress(5, LIMITS)
    progress.feed(" NS 1-5s 2.527e+01  [-33 2.043e-02 2.043e-02 3.566e-01]  [0 -10 32]")
    assert progress.finish(0).errors == ["svMultiPhysics ended after 1 of 5 steps."]
    assert progress.finish(0, stopped=True).errors == [], "stopped is what was asked for"
    assert "exit code 139" in progress.finish(139).errors[0]


def test_the_test_is_the_cases_own_file_with_its_steps_and_saving_changed_beside_it(tmp_path):
    case_dir = tmp_path / "aorta"
    case.write_case(case_dir, testing.cube_mesh(), NAMES, CONDITIONS)
    before = (case_dir / "solver.xml").read_bytes()
    stale = tmp_path / "aorta-test" / "1-procs" / "B_NS_Velocity_flux.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("a previous test's rows\n")

    path = trial.write_trial(case_dir, steps=3)
    assert path == tmp_path / "aorta-test" / "solver.xml"
    assert (case_dir / "solver.xml").read_bytes() == before, "the case is left as it is"
    assert not stale.exists(), "svMultiPhysics appends to a face results file it finds"
    root = ET.parse(path).getroot()
    general = root.find("GeneralSimulationParameters")
    assert general.findtext("Number_of_time_steps") == "3"
    assert general.findtext("Increment_in_saving_VTK_files") == "3", "the last step, timed as the run's"
    assert general.findtext("Increment_in_saving_restart_files") == "3"
    assert general.findtext("Searched_file_name_to_trigger_stop") == "STOP_SIM"
    mesh = root.findtext("Add_mesh/Mesh_file_path")
    assert mesh == "../aorta/mesh/mesh-complete.mesh.vtu"
    for element in root.iter():
        if element.tag.endswith("file_path"):
            assert (path.parent / element.text).is_file(), element.text
    # Everything else as written: the conditions, the scale, the equation.
    assert root.findtext("Add_mesh/Mesh_scale_factor") == "0.1"
    assert root.find("Add_equation[@type='fluid']/Add_BC[@name='cap_outlet']/RCR_values") is not None
    assert trial.schedule(case_dir).steps == 10000


def test_the_full_runs_saves_are_counted_as_the_solver_counts_them():
    assert LIMITS.saves == 100
    assert trial.Schedule(10, 3, 1, 1, 5, 1e-3).saves == 3       # steps 3, 6, 9
    assert trial.Schedule(10, 1, 4, 1, 5, 1e-3).saves == 7       # steps 4 to 10


def test_the_solver_is_the_one_asked_for_else_the_variable_else_the_path_else_the_newest_installed(tmp_path):
    root = tmp_path / "sv"
    old = executable(root / "2025-06-20" / "bin" / "svmultiphysics")
    new = executable(root / "2026-06-11" / "bin" / "svmultiphysics")
    on_path = executable(tmp_path / "bin" / "svmultiphysics")
    mine = executable(tmp_path / "build" / "svMultiPhysics-build" / "bin" / "svmultiphysics")
    mpiexec = executable(tmp_path / "bin" / "mpiexec")
    env = {"PATH": str(tmp_path / "bin")}

    assert trial.installed_solvers(root) == [str(old), str(new)]
    assert trial.find_solver(None, {}, install_root=root, platform="linux").executable == str(new)
    assert trial.find_solver(None, env, install_root=root, platform="linux").executable == str(on_path)
    by_variable = trial.find_solver(None, dict(env, SVMULTIPHYSICS=str(mine)), install_root=root,
                                    platform="linux")
    assert by_variable.executable == str(mine) and by_variable.mpiexec == str(mpiexec)
    assert trial.find_solver(str(mine), env, install_root=root, platform="linux").executable == str(mine)
    assert trial.find_solver(str(tmp_path / "gone"), env, install_root=root, platform="linux") is None, \
        "a build asked for and not there is not replaced by another"


def test_on_windows_it_is_found_in_wsl_as_on_linux():
    printed = ("installed:/usr/local/sv/svMultiPhysics/2025-06-20/bin/svmultiphysics\n"
               "installed:/usr/local/sv/svMultiPhysics/2026-06-11/bin/svmultiphysics\n"
               "mpiexec:/usr/bin/mpiexec\n")
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, printed, "")

    found = trial.find_solver(None, {}, install_root="/nowhere", platform="win32", wsl_runner=runner)
    assert found == trial.Solver("/usr/local/sv/svMultiPhysics/2026-06-11/bin/svmultiphysics",
                                 "/usr/bin/mpiexec", wsl=True)
    assert calls[0][:3] == ["wsl.exe", "--exec", "sh"]
    assert trial.parse_wsl_probe(printed + "path:/home/me/bin/svmultiphysics\n").executable \
        == "/home/me/bin/svmultiphysics", "its PATH first, as on Linux"
    assert trial.parse_wsl_probe("") is None


def test_the_command_is_mpiexec_with_the_processes_else_the_solver_alone(tmp_path):
    native = trial.Solver("/sv/bin/svmultiphysics", "/usr/bin/mpiexec")
    assert trial.command(native, 8, tmp_path) == ["/usr/bin/mpiexec", "-np", "8", "/sv/bin/svmultiphysics", "solver.xml"]
    assert trial.command(native, 1, tmp_path) == ["/sv/bin/svmultiphysics", "solver.xml"]
    assert trial.command(trial.Solver("/sv/bin/svmultiphysics"), 8, tmp_path) == ["/sv/bin/svmultiphysics", "solver.xml"]
    wsl = trial.Solver("/usr/local/sv/svMultiPhysics/2026-06-11/bin/svmultiphysics", "/usr/bin/mpiexec", wsl=True)
    assert trial.command(wsl, 4, r"C:\cases\aorta-test") == [
        "wsl.exe", "--cd", r"C:\cases\aorta-test", "--exec", "/usr/bin/mpiexec", "-np", "4",
        "/usr/local/sv/svMultiPhysics/2026-06-11/bin/svmultiphysics", "solver.xml"]


FLUX = """\
# svMultiPhysics boundary integral results file.
# Format
face areas:  1.200e+02  2.709e+00  4.513e-01
step   time   wall  cap_inlet  cap_outlet
 1   1.0000000000e-03  0.0000000000e+00  -5.0200000000e+00  1.9271432769e+00
 2   2.0000000000e-03  0.0000000000e+00  -5.0500000000e+00  -4.7683930961e+00
"""


def test_the_flow_through_each_face_is_turned_round_to_in_positive_and_held_to_its_condition(tmp_path):
    path = tmp_path / "B_NS_Velocity_flux.txt"
    path.write_text(FLUX)
    integrals = trial.read_face_integrals(path)
    assert integrals.names == ("wall", "cap_inlet", "cap_outlet")
    assert integrals.areas == (120.0, 2.709, 0.4513)

    first = {flow.name: flow for flow in trial.face_flows(integrals, CONDITIONS, step=1)}
    assert first["cap_inlet"].flow == pytest.approx(5.02) and first["cap_inlet"].right_way
    assert first["cap_inlet"].waveform == pytest.approx(5.025)
    assert first["cap_outlet"].flow == pytest.approx(-1.9271432769) and first["cap_outlet"].right_way
    lines = trial.flow_lines(list(first.values()), 1e-3)
    assert lines[0] == "At 0.001 s: the flow comes in through every inflow and leaves through every outlet."

    last = trial.face_flows(integrals, CONDITIONS)
    lines = trial.flow_lines(last, 2e-3)
    assert lines[0] == "At 0.002 s: the flow goes the wrong way through cap_outlet."
    assert lines[1] == "cap_outlet: 4.77 mL/s in -- it should flow out"


def test_the_arrows_point_along_the_velocity_and_carry_its_speed():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    import numpy as np

    grid = testing.cube_mesh(4)
    velocity = np.tile([0.0, 0.0, 10.0], (grid.GetNumberOfPoints(), 1))
    array = numpy_to_vtk(velocity, deep=True)
    array.SetName("Velocity")
    grid.GetPointData().AddArray(array)
    arrows = trial.velocity_arrows(grid, count=20)
    assert arrows.GetNumberOfPoints() > 0
    speed = vtk_to_numpy(arrows.GetPointData().GetArray("Speed"))
    assert speed == pytest.approx(10.0)
    one = trial.velocity_arrows(grid, count=1).GetBounds()
    assert one[5] - one[4] > 3 * (one[1] - one[0]), "one arrow, drawn along the flow, in z"
    with pytest.raises(ValueError):
        trial.velocity_arrows(testing.cube_mesh(2))


SOLVER = os.environ.get("SVMULTIPHYSICS") or shutil.which("svmultiphysics")


@pytest.mark.skipif(not SOLVER, reason="svmultiphysics is not on the PATH or at SVMULTIPHYSICS")
def test_the_cube_steps_through_its_test_and_the_flow_goes_the_right_way(tmp_path):
    case_dir = tmp_path / "cube"
    case.write_case(case_dir, testing.cube_mesh(), NAMES, CONDITIONS)
    path = trial.write_trial(case_dir, steps=2)
    solver = trial.Solver(SOLVER)
    completed = subprocess.run(trial.command(solver, 1, path.parent), cwd=path.parent,
                               capture_output=True, text=True, timeout=600)
    progress = trial.Progress(2, trial.schedule(case_dir))
    for line in (completed.stdout + completed.stderr).splitlines():
        progress.feed(line)
    report = progress.finish(completed.returncode)
    assert report.ok, report.errors + list(progress.tail)[-20:]
    results = trial.results_folder(path.parent, 1)
    assert [p.name for p in trial.result_files(results)] == ["result_002.vtu"], "the last step only"
    flows = trial.face_flows(trial.read_face_integrals(results / trial.FLUX_FILE), CONDITIONS)
    assert all(flow.right_way for flow in flows), [flow.describe() for flow in flows]
    arrows = trial.velocity_arrows(trial.read_result(trial.result_files(results)[-1]), count=100)
    assert arrows.GetNumberOfPoints() > 0
