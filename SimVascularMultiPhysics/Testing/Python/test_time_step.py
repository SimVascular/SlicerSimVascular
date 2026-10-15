"""The time step a CFL number asks for: the units, the profile, the mesh size and the rounding."""

import math

import pytest
import vtk

from svmpsetup import settings, time_step
from svromsetup.boundary_conditions import RCR, Inflow, Resistance

# 100 mm^2 is 1 cm^2 at the default scale factor, so a flow in mL/s is a mean velocity in cm/s.
AREAS = {"cap_inlet": 100.0, "cap_outlet": 50.0}
WAVEFORM = Inflow((0.0, 0.2, 0.8), (5.0, 20.0, 5.0), "/somewhere/cap_inlet.flow")


def values(**changed):
    resolved = settings.resolve()
    resolved.update(changed)
    return resolved


def test_a_flow_in_ml_per_s_over_a_cap_in_mm2_is_a_velocity_in_cm_per_s():
    flat = time_step.fastest_inflow({"cap_inlet": Inflow.steady(10.0)}, AREAS, "Flat", 0.1)
    assert (flat.flow, flat.area, flat.mean_velocity, flat.peak_velocity) == pytest.approx((10.0, 100.0, 10.0, 10.0))
    parabolic = time_step.fastest_inflow({"cap_inlet": Inflow.steady(10.0)}, AREAS, "Parabolic", 0.1)
    assert parabolic.peak_velocity == pytest.approx(20.0), "a Poiseuille profile peaks at twice its mean"
    unscaled = time_step.fastest_inflow({"cap_inlet": Inflow.steady(10.0)}, AREAS, "Flat", 1.0)
    assert unscaled.mean_velocity == pytest.approx(0.1), "the area goes with the square of the factor"


def test_the_fastest_inflow_is_its_waveforms_peak_on_the_inflow_where_that_is_fastest():
    conditions = {"cap_inlet": WAVEFORM, "cap_outlet": Inflow.steady(15.0), "cap_other": Resistance(1.0)}
    fastest = time_step.fastest_inflow(conditions, AREAS, "Flat", 0.1)
    assert fastest.name == "cap_outlet", "15 mL/s through 0.5 cm^2 beats 20 through 1"
    assert fastest.peak_velocity == pytest.approx(30.0)
    with pytest.raises(time_step.EstimateError, match="no inflow"):
        time_step.fastest_inflow({"cap_outlet": RCR(1.0, 1e-4, 10.0)}, AREAS)


def single_tetrahedron(with_face=False):
    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        points.InsertNextPoint(point)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_TETRA, 4, (0, 1, 2, 3))
    if with_face:
        grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, (0, 1, 2))
    return grid


def test_the_mean_edge_length_counts_each_edge_once_whatever_shares_it():
    expected = (3 * 1.0 + 3 * math.sqrt(2)) / 6
    assert time_step.mean_edge_length(single_tetrahedron()) == pytest.approx(expected)
    assert time_step.mean_edge_length(single_tetrahedron(with_face=True)) == pytest.approx(expected), \
        "a labelled mesh's face triangles do not count their edges again"


def test_the_step_is_rounded_down_to_whole_steps_per_cycle_and_the_cycles_are_kept():
    # 20 mL/s through 1 cm^2, twice that on the centreline: 40 cm/s. Edges of 1 mm, 0.1 cm: at
    # CFL 1, 0.1 / 40 = 2.5e-3 s, so a 0.8 s cycle is 320 steps exactly. Saving every 100 steps of
    # 1 ms is 8 results a 0.8 s cycle, which 320 steps divide: every 40.
    exact = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=12)
    assert exact.cfl_time_step == pytest.approx(2.5e-3)
    assert (exact.steps_per_cycle, exact.cycles) == (320, 12)
    assert exact.steps == 12 * 320 and exact.time_step == pytest.approx(2.5e-3)
    assert exact.previous_cfl == pytest.approx(0.4), "1e-3 s was CFL 0.4"
    assert exact.saving == time_step.Saving(results_every=40, start_after=1, restart_every=40)
    assert exact.results_interval == pytest.approx(0.1), "results as often in time as before"

    # At CFL 0.9, 355.6 steps a cycle round up to 356, and up again to 360, a multiple of the 8
    # results a cycle: every 45 steps lands at the same point of every cycle.
    under = time_step.estimate(0.9, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values())
    assert under.steps_per_cycle == 360 and under.saving.results_every == 45
    assert under.time_step == pytest.approx(0.8 / 360)
    assert under.time_step <= under.cfl_time_step, "never a CFL above the one asked for"


def test_a_steady_inflow_has_its_step_to_three_figures_and_its_steps_left_as_they_are():
    found = time_step.estimate(1.0, 0.7, {"cap_inlet": Inflow.steady(10.0)}, AREAS, values(), cycles=10)
    # 20 cm/s on the centreline, 0.07 cm: 3.5e-3 s, kept to three figures. No cycle to count.
    assert found.period is None and found.time_step == pytest.approx(3.5e-3)
    assert (found.cycles, found.steps) == (None, 10000)
    assert found.saving.results_every == round(0.1 / 3.5e-3), "every 0.1 s, as near as steps go"
    assert time_step.describe(found).startswith(
        "Steady inflow 10 mL/s through cap_inlet (100 mm²): 10 cm/s across the cap, 20 cm/s on its centreline"), \
        "a steady inflow's one flow is its peak, and is called what it is"


def test_saving_that_starts_later_or_comes_less_than_once_a_cycle_is_kept_in_time():
    # Saving from step 5001, 5 s in, starts 5 s in at the new step too; once a second, with a
    # 0.8 s cycle, is once a cycle, the nearest whole number of them.
    later = values(**{"general.start_saving_after_time_step": 5001,
                      "general.increment_in_saving_vtk_files": 1000,
                      "general.increment_in_saving_restart_files": 500})
    found = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, later)
    assert found.saving.start_after == 1 + round(5.0 / 2.5e-3)
    assert found.saving.results_every == 320, "every whole cycle"
    assert found.saving.restart_every == round(0.5 / 2.5e-3)


def test_what_cannot_be_estimated_is_refused():
    with pytest.raises(time_step.EstimateError, match="Impose flux"):
        time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(**{"bc.impose_flux": False}))
    with pytest.raises(time_step.EstimateError, match="more than 0"):
        time_step.estimate(0.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values())
    with pytest.raises(time_step.EstimateError, match="area"):
        time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, {}, values())


def test_the_estimate_is_said_with_its_units_and_what_it_does_to_the_saving():
    found = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=12)
    text = time_step.describe(found)
    assert "Peak inflow 20 mL/s through cap_inlet (100 mm²): 20 cm/s across the cap, 40 cm/s on its centreline" in text
    assert "Mean edge length 1 mm, 0.1 cm at the mesh scale factor 0.1." in text
    assert "a whole 320 steps per 0.8 s cycle by rounding down, so the CFL is at most 1. 12 cycles: 3840 steps." in text
    assert "was CFL 0.4" in text
    assert ("Results are saved every 40 steps, 0.1 s, 8 a cycle: 96 files over the run, as often in "
            "time as before, where every 100 steps would have been 38.") in text
    # Worked out again with nothing changed, it says nothing was replaced.
    again = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS,
                               values(**{"general.time_step_size": found.time_step,
                                         "general.increment_in_saving_vtk_files": 40,
                                         "general.increment_in_saving_restart_files": 40}), cycles=12)
    said = time_step.describe(again)
    assert "replaces" not in said and "as often in time as before" not in said and "Restart" not in said
    assert "Restart files are written every 40 steps, as often in time as before." in text


def test_a_run_set_in_cardiac_cycles_is_cycles_times_the_steps_in_one():
    assert time_step.DEFAULT_CYCLES == 10
    assert time_step.steps_for_cycles(10, {"cap_inlet": WAVEFORM}, 1e-3) == 8000      # 10 x 0.8 s
    assert time_step.steps_for_cycles(10, {"cap_inlet": WAVEFORM}, 3e-3) == round(8.0 / 3e-3)
    assert time_step.steps_for_cycles(10, {"cap_inlet": Inflow.steady(10.0)}, 1e-3) is None, \
        "a steady inflow has no cycle to count"
    # The estimate calculates the steps from the cycles it is given; with none, the number of
    # steps is user-defined and left as it is, whatever time it now runs.
    found = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=10)
    assert (found.cycles, found.steps) == (10, 10 * 320)
    user = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values())
    assert (user.cycles, user.steps) == (None, 10000)
    assert "The number of time steps is user-defined: its 10000 steps now run 25 s." in time_step.describe(user)


def test_a_max_velocity_or_mesh_size_set_by_hand_is_used_and_said_beside_the_cases():
    # 80 cm/s by hand, twice the inflow's 40: half the step, 640 steps a 0.8 s cycle.
    fast = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=12,
                              velocity=80.0, mesh_edge_length=1.0)
    assert fast.cfl_time_step == pytest.approx(1.25e-3) and fast.steps_per_cycle == 640
    text = time_step.describe(fast)
    assert text.startswith("Max velocity 80 cm/s, set by hand; the case gives 40 cm/s, from its peak inflow "
                           "20 mL/s through cap_inlet (100 mm²)")
    assert "Mean edge length 1 mm" in text, "the mesh's own size, not set by hand"

    fine = time_step.estimate(1.0, 0.5, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=12,
                              mesh_edge_length=1.0)
    assert fine.cfl_time_step == pytest.approx(1.25e-3)
    assert "Mesh size 0.5 mm, set by hand, 0.05 cm at the mesh scale factor 0.1; the mesh's mean edge " \
           "length is 1 mm." in time_step.describe(fine)


def test_a_velocity_set_by_hand_needs_no_inflow_but_has_to_be_more_than_0():
    found = time_step.estimate(1.0, 1.0, {"cap_outlet": RCR(1.0, 1e-4, 10.0)}, AREAS, values(), velocity=50.0)
    assert found.speed is None and found.time_step == pytest.approx(2e-3)
    assert time_step.describe(found).startswith("Max velocity 50 cm/s, set by hand.")
    with pytest.raises(time_step.EstimateError, match="max velocity"):
        time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), velocity=0.0)
    with pytest.raises(time_step.EstimateError, match="mesh size"):
        time_step.estimate(1.0, 0.0, {"cap_inlet": WAVEFORM}, AREAS, values())
    # With Impose flux off the inflow gives no velocity, but one set by hand still estimates.
    unflux = values(**{"bc.impose_flux": False})
    assert time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS, unflux, velocity=40.0).speed is None


def test_a_saving_set_by_hand_is_kept_as_set_and_the_step_only_made_whole_per_cycle():
    # At CFL 0.9, 355.6 steps a cycle: adapted, the saving would make them 360, 8 results of 45
    # steps. Set by hand it is left at every 100, and the steps per cycle only made whole, 356.
    found = time_step.estimate(0.9, 1.0, {"cap_inlet": WAVEFORM}, AREAS, values(), cycles=12, adapt_saving=False)
    assert found.saving == time_step.Saving(results_every=100, start_after=1, restart_every=100)
    assert found.steps_per_cycle == 356 and found.time_step == pytest.approx(0.8 / 356)
    text = time_step.describe(found)
    assert ("The saving is as set: results every 100 steps, 0.225 s, 42 files over the run, 3.56 a cycle of "
            "356 steps, so at a different point of each") in text
    assert "as often in time as before" not in text and "The time step it replaces, 0.001 s" in text
    # One that divides the cycle is said to, and one longer than it in whole cycles.
    on_cycle = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS,
                                  values(**{"general.increment_in_saving_vtk_files": 40}), adapt_saving=False)
    assert on_cycle.steps_per_cycle == 320 and "8 a cycle at the same points of each" in time_step.describe(on_cycle)
    every_other = time_step.estimate(1.0, 1.0, {"cap_inlet": WAVEFORM}, AREAS,
                                     values(**{"general.increment_in_saving_vtk_files": 640}), adapt_saving=False)
    assert "once every 2 cycles" in time_step.describe(every_other)
    # A steady inflow's step to three figures, as adapted, and the saving as set.
    steady = time_step.estimate(1.0, 0.7, {"cap_inlet": Inflow.steady(10.0)}, AREAS,
                                values(**{"general.start_saving_after_time_step": 5001}), adapt_saving=False)
    assert steady.time_step == pytest.approx(3.5e-3)
    assert steady.saving == time_step.Saving(results_every=100, start_after=5001, restart_every=100)
