"""The time step a CFL number asks for, from the fastest inflow and the size of the mesh.

    dt = CFL * h / u

with `h` the mesh's mean edge length and `u` the fastest velocity an inflow imposes, both in the
solver's units. A step much longer than that lets the flow cross several elements in one step,
which costs accuracy and nonlinear iterations, and can make a run diverge; one much shorter costs
time for nothing. 1 is the CFL recommended for svMultiPhysics, and the default here.

**The units are where this goes wrong without a word.** An inflow is a flow, in mL/s -- cm^3/s,
the solver's units already -- but the mesh is in millimetres, as Slicer's coordinates are, and
reaches the solver through `Mesh_scale_factor`. So the cap's area, in mm^2, is scaled by the
factor squared to cm^2 before a velocity is taken from it, and the edge length, in mm, by the
factor to cm. Through the factor the case is written with rather than a fixed 0.1, so the estimate
is for the mesh the solver will read: a case written with the factor changed is estimated as it
will run.

**The velocity is the fastest the solver imposes**, which with the default Parabolic profile is
on the cap's centreline: twice the mean velocity across it, since a Poiseuille profile peaks at
twice its mean. With a Flat profile the two are one. The mean -- peak flow over area -- would give
a step twice too long for the default profile. With Impose flux off an inflow's values are
velocities rather than flows, and nothing here can say what the peak of one is; that case is
refused rather than estimated as though it were a flow.

**The mesh size is the mean edge length**, the element size the mesh was made at, near enough:
CFD Mesh Generator's 1 mm target gives a mean of about 0.73 mm. Rejected, as alternatives:
- *the smallest edge*, which is set by the worst few elements of a million -- on the tutorial's
  aorta, 0.28 mm against a mean of 0.73 -- and would ask for a step two and a half times shorter
  than the rest of the mesh needs;
- *the edges at the inflow cap*, which is where the inflow is fastest, but not where the flow is
  fastest in a model with a narrowing downstream.

**The step is rounded down to a whole number of steps per cardiac cycle**, and the number of steps
set to keep the cycles the case was set to run, so that a run ends on a cycle's end. Rounded down,
never up: the CFL that comes out is at most the one asked for.

**The saving is kept as it was in time, not in steps.** Every saving setting counts steps, so a
step five times shorter at the same "every 100 steps" saves five times as many results -- 40 MB
each on a million elements -- and "start saving after step 5000" starts at a fifth of the time it
did. So the results are saved as often per cycle as they were (every 0.1 s in a 0.91 s cycle is
nine a cycle), with the steps per cycle rounded up to a multiple of that so each lands at the same
point of every cycle -- a saving interval that does not divide the cycle drifts through it, and no
two cycles can be compared result for result. Rounding the steps up shortens the step, so the CFL
still does not exceed the one asked for. Saving is started at the same time it was, and restart
files written as often in time as they were; those need not fall on the cycle.

**Or the saving is the user's, as set**, with `adapt_saving` off: kept in steps exactly as given,
and the step rounded only to whole steps per cycle, since there is no saving to fit it to. What it
does then -- results that fall at a different point of each cycle, more files for a shorter step --
is said rather than corrected: someone who typed "every 1000 steps" meant it, and has it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import vtk

from svromsetup import boundary_conditions as bcs

DEFAULT_CFL = 1.0
# How many cardiac cycles a run is set to by default, when its length is set in cycles, the lab's
# choice: the RCR outlets start empty and take the first few cycles to fill, which leaves cycles
# after them to read results from.
DEFAULT_CYCLES = 10
# A Poiseuille profile's peak over its mean.
PARABOLIC_PEAK_OVER_MEAN = 2.0


class EstimateError(ValueError):
    """Raised when the case does not say enough to estimate a time step from."""


@dataclass(frozen=True)
class InflowSpeed:
    """The fastest inflow: its cap, its peak flow (mL/s), its cap's area (mm^2), and the mean and
    imposed peak velocity across that cap at that flow (cm/s); `steady` when it has one flow
    throughout, which then is its peak."""

    name: str
    flow: float
    area: float
    mean_velocity: float
    peak_velocity: float
    steady: bool = False


def fastest_inflow(conditions_by_name, cap_areas, profile="Parabolic", scale_factor=0.1) -> InflowSpeed:
    """The inflow whose peak imposes the fastest velocity, over every inflow of the case.

    `cap_areas` is {cap name: area in the mesh's units, mm^2}; `scale_factor` what the mesh is
    multiplied by to reach the solver's centimetres, so an area is multiplied by its square.
    """
    peak_over_mean = PARABOLIC_PEAK_OVER_MEAN if profile == "Parabolic" else 1.0
    fastest = None
    for name, condition in conditions_by_name.items():
        if not isinstance(condition, bcs.Inflow):
            continue
        area = cap_areas.get(name)
        if not area or area <= 0:
            raise EstimateError(f"The area of {name} is not known, so no velocity can be taken from its flow.")
        flow = float(np.max(np.abs(condition.flow)))
        mean = flow / (area * scale_factor ** 2)
        speed = InflowSpeed(name, flow, float(area), mean, mean * peak_over_mean, condition.is_steady)
        if fastest is None or speed.peak_velocity > fastest.peak_velocity:
            fastest = speed
    if fastest is None:
        raise EstimateError("There is no inflow to take a velocity from: give a cap an Inflow condition.")
    if fastest.peak_velocity <= 0:
        raise EstimateError(f"{fastest.name}'s inflow is zero throughout, so no time step follows from it.")
    return fastest


def mean_edge_length(mesh) -> float:
    """The mean length of the mesh's edges, in its own units: each edge once, however many cells
    share it, so that the faces' triangles a labelled mesh carries do not count their edges again."""
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(mesh)
    edges.UseAllPointsOn()
    edges.Update()
    lines = edges.GetOutput()
    if lines.GetNumberOfLines() == 0:
        raise EstimateError("The mesh has no edges to measure.")
    from vtk.util.numpy_support import vtk_to_numpy

    ends = vtk_to_numpy(lines.GetLines().GetConnectivityArray()).reshape(-1, 2)
    points = vtk_to_numpy(lines.GetPoints().GetData())
    return float(np.linalg.norm(points[ends[:, 0]] - points[ends[:, 1]], axis=1).mean())


@dataclass(frozen=True)
class Saving:
    """When results and restart files are written, in steps: what the solver is told."""

    results_every: int
    start_after: int
    restart_every: int

    def results(self, steps) -> int:
        """How many result files a run of `steps` writes: a step is saved when it is a multiple
        of the increment, from the start step on."""
        first = max(self.start_after, 1)
        return max(0, steps // self.results_every - (first - 1) // self.results_every)


@dataclass(frozen=True)
class Estimate:
    cfl: float
    edge_length: float           # mm, the mesh's own units: the mesh size the step is for
    speed: InflowSpeed | None    # what the inflow gives; None when there is none and a velocity was set
    scale_factor: float
    cfl_time_step: float         # s: what the CFL asks for, before rounding
    time_step: float             # s: rounded down to whole steps per cycle, and per saving
    period: float | None         # s: the inflows' cardiac cycle, None for a steady inflow
    steps_per_cycle: int | None
    cycles: int | None           # what the steps were calculated from; None when user-defined
    steps: int
    saving: Saving
    previous_time_step: float
    previous_steps: int
    previous_saving: Saving
    velocity: float = 0.0        # cm/s: the max velocity the step is for
    velocity_by_hand: bool = False
    mesh_edge_length: float | None = None   # mm: the mesh's own mean edge length, beside one set by hand
    saving_adapted: bool = True  # False: the saving is the user's, left in steps as set

    @property
    def previous_cfl(self) -> float:
        return self.cfl * self.previous_time_step / self.cfl_time_step

    @property
    def achieved_cfl(self) -> float:
        return self.cfl * self.time_step / self.cfl_time_step

    @property
    def simulated_time(self) -> float:
        return self.steps * self.time_step

    @property
    def results_interval(self) -> float:
        return self.saving.results_every * self.time_step


def _period(conditions_by_name) -> float | None:
    """The cardiac cycle, by the rule the conditions are written with (`cycle_period`: the first
    waveform's); None when every inflow is steady, which has no cycle to keep whole -- not the
    1 s a steady inflow is spread over when it is written."""
    conditions = list(conditions_by_name.values())
    if not any(isinstance(condition, bcs.Inflow) and not condition.is_steady for condition in conditions):
        return None
    return bcs.cycle_period(conditions)


def _round_down(value, digits=3) -> float:
    """`value` to `digits` significant figures, never up."""
    if value <= 0:
        return value
    exponent = math.floor(math.log10(value)) - digits + 1
    return math.floor(value / 10 ** exponent) * 10 ** exponent


def has_waveform(conditions_by_name) -> bool:
    """Whether any inflow is a waveform: a steady inflow has no cardiac cycle to count."""
    return any(isinstance(condition, bcs.Inflow) and not condition.is_steady
               for condition in conditions_by_name.values())


def steps_for_cycles(cycles, conditions_by_name, step) -> int | None:
    """The number of steps that runs `cycles` cardiac cycles at `step`, or None when there is no
    waveform to have a cycle. The nearest whole number: exactly cycles times steps per cycle
    when the step divides the cycle, as the CFL estimate makes it."""
    period = _period(conditions_by_name)
    if period is None or not step > 0:
        return None
    return max(1, round(cycles * period / step))


def _same_time(steps, old_step, new_step) -> int:
    """A count of old steps as the same time in new ones, a whole number of them, at least 1."""
    return max(1, round(steps * old_step / new_step))


def estimate(cfl, edge_length, conditions_by_name, cap_areas, values, cycles=None, velocity=None,
             mesh_edge_length=None, adapt_saving=True) -> Estimate:
    """The time step, number of steps and saving that run the case's cycles at `cfl`.

    `values` are the panel's settings, resolved: the inflow profile, Impose flux, the mesh scale
    factor, and the time stepping and saving as they stand. The saving is kept in time -- the
    results as many per cycle -- or, for a steady inflow, as near as whole steps go. `cycles`,
    given, is how many cardiac cycles the run is calculated from, and the steps are that many
    cycles; not given, the number of steps is user-defined, and is left as it is, whatever length
    of time it now runs. A steady inflow has no cycles, and its steps are left as they are.

    `edge_length` is the mesh size the step is for, in mm: the mesh's mean edge length, or one set
    by hand, beside which `mesh_edge_length` is the mesh's own, to say both. `velocity`, given, is a
    max velocity set by hand, in cm/s, used in place of the inflow's -- which is still worked out,
    where there is one, to say beside it.

    `adapt_saving` off, the saving is left as it is, in steps, and the step is rounded down to whole
    steps per cycle alone: there is no saving to fit it to.
    """
    if not cfl > 0:
        raise EstimateError("The CFL number has to be more than 0.")
    if not edge_length > 0:
        raise EstimateError("The mesh size has to be more than 0.")
    scale = values["mesh.scale_factor"]
    try:
        if not values["bc.impose_flux"]:
            raise EstimateError("With Impose flux off, an inflow's values are velocities rather than flows, "
                                "and no peak velocity can be taken from them. Turn Impose flux on, or set "
                                "the max velocity by hand.")
        speed = fastest_inflow(conditions_by_name, cap_areas, values["bc.inflow_profile"], scale)
    except EstimateError:
        if velocity is None:
            raise
        speed = None
    if velocity is not None and not velocity > 0:
        raise EstimateError("The max velocity has to be more than 0.")
    used = velocity if velocity is not None else speed.peak_velocity
    cfl_step = cfl * edge_length * scale / used
    old_step = values["general.time_step_size"]
    old_steps = values["general.number_of_time_steps"]
    old_saving = Saving(values["general.increment_in_saving_vtk_files"],
                        values["general.start_saving_after_time_step"],
                        values["general.increment_in_saving_restart_files"])
    interval = old_saving.results_every * old_step     # s between results, as the case stands
    period = _period(conditions_by_name)
    if period:
        per_cycle = math.ceil(period / cfl_step - 1e-9)
        if not adapt_saving:
            results_every = old_saving.results_every
        elif interval <= period:
            # As many results per cycle as there were, on steps that divide the cycle.
            per_save_cycle = max(1, round(period / interval))
            per_cycle = math.ceil(per_cycle / per_save_cycle) * per_save_cycle
            results_every = per_cycle // per_save_cycle
        else:
            # Fewer than one a cycle: every so many whole cycles, as before.
            results_every = per_cycle * max(1, round(interval / period))
        step = period / per_cycle
        cycles = max(1, int(cycles)) if cycles else None
        steps = cycles * per_cycle if cycles else old_steps
    else:
        per_cycle = cycles = None
        step = _round_down(cfl_step)
        steps = old_steps
        results_every = (_same_time(old_saving.results_every, old_step, step) if adapt_saving
                         else old_saving.results_every)
    if adapt_saving:
        # Saving starts when it did, in time: from the first step, if it did; and restart files as
        # often in time as they were.
        start = old_saving.start_after if old_saving.start_after <= 1 else \
            1 + round((old_saving.start_after - 1) * old_step / step)
        saving = Saving(results_every, start, _same_time(old_saving.restart_every, old_step, step))
    else:
        saving = old_saving
    return Estimate(cfl, edge_length, speed, scale, cfl_step, step, period, per_cycle, cycles, steps,
                    saving, old_step, old_steps, old_saving, used, velocity is not None,
                    mesh_edge_length, adapt_saving)


def describe(estimate: Estimate) -> str:
    """The estimate in words, with every number's units, and what it did to the saving."""
    speed, scale = estimate.speed, estimate.scale_factor
    text = []
    if speed is not None:
        profile = (f"{speed.peak_velocity:.3g} cm/s on its centreline with the Parabolic profile"
                   if speed.peak_velocity != speed.mean_velocity else "the same at every point with the Flat profile")
        inflow = (f"{'steady' if speed.steady else 'peak'} inflow {speed.flow:.4g} mL/s through {speed.name} "
                  f"({speed.area:.4g} mm²): "
                  f"{speed.mean_velocity:.3g} cm/s across the cap, {profile}")
    if not estimate.velocity_by_hand:
        text.append(inflow[0].upper() + inflow[1:] + ".")
    else:
        text.append(f"Max velocity {estimate.velocity:.4g} cm/s, set by hand"
                    + (f"; the case gives {speed.peak_velocity:.3g} cm/s, from its {inflow}" if speed else "") + ".")
    edge, mesh = estimate.edge_length, estimate.mesh_edge_length
    if mesh is None or math.isclose(edge, mesh, rel_tol=1e-9):
        text.append(f"Mean edge length {edge:.3g} mm, {edge * scale:.3g} cm at the mesh scale factor {scale:g}.")
    else:
        text.append(f"Mesh size {edge:.4g} mm, set by hand, {edge * scale:.3g} cm at the mesh scale factor "
                    f"{scale:g}; the mesh's mean edge length is {mesh:.3g} mm.")
    # The step once: rounding to whole steps per cycle moves it in a figure nobody reads, and the
    # exact value is in its field.
    if estimate.period:
        text.append(f"At CFL {estimate.cfl:g}: {estimate.cfl_time_step:.4g} s, made a whole "
                    f"{estimate.steps_per_cycle} steps per {estimate.period:.4g} s cycle by rounding down, so "
                    f"the CFL is at most {estimate.cfl:g}.")
    else:
        text.append(f"At CFL {estimate.cfl:g}: {estimate.cfl_time_step:.5g} s, rounded down to "
                    f"{estimate.time_step:.3g} s.")
    if estimate.cycles:
        text.append(f"{estimate.cycles} cycle{'s' if estimate.cycles != 1 else ''}: {estimate.steps} steps.")
    else:
        text.append(f"The number of time steps is user-defined: its {estimate.steps} steps now run "
                    f"{estimate.simulated_time:.3g} s.")
    # What it replaced, only where it replaced something: worked out again as the panel changes,
    # the estimate is mostly of a step it set itself, and "it replaces 0.0002 s" said of 0.0002 s
    # reads as a change that was not.
    if not math.isclose(estimate.previous_time_step, estimate.time_step, rel_tol=1e-9):
        text.append(f"The time step it replaces, {estimate.previous_time_step:g} s, was CFL "
                    f"{estimate.previous_cfl:.2g}.")
    saving, before = estimate.saving, estimate.previous_saving
    if not estimate.saving_adapted:
        text.append(_saving_as_set(estimate))
        return " ".join(text)
    per = (f", {round(estimate.period / estimate.results_interval)} a cycle"
           if estimate.period and estimate.results_interval <= estimate.period else "")
    sentence = (f"Results are saved every {saving.results_every} steps, {estimate.results_interval:.3g} s{per}: "
                f"{saving.results(estimate.steps)} files over the run")
    if saving.results_every != before.results_every:
        sentence += (f", as often in time as before, where every {before.results_every} steps would have "
                     f"been {before.results(estimate.steps)}")
    text.append(sentence + ".")
    moved = []
    if saving.start_after != before.start_after:
        moved.append(f"saving starts after step {saving.start_after}, at the time step {before.start_after} was")
    if saving.restart_every != before.restart_every:
        moved.append(f"restart files are written every {saving.restart_every} steps, as often in time as before")
    if moved:
        text.append(moved[0][0].upper() + moved[0][1:] + ("; " + "; ".join(moved[1:]) if moved[1:] else "") + ".")
    return " ".join(text)


def _saving_as_set(estimate: Estimate) -> str:
    """The saving the user set, in words: what it comes to at this step, and whether results fall at
    the same point of every cycle -- said, not corrected, since it was set by hand."""
    saving, per_cycle = estimate.saving, estimate.steps_per_cycle
    every = saving.results_every
    sentence = (f"The saving is as set: results every {every} steps, {estimate.results_interval:.3g} s, "
                f"{saving.results(estimate.steps)} files over the run")
    if per_cycle:
        if per_cycle % every == 0:
            sentence += f", {per_cycle // every} a cycle at the same points of each"
        elif every % per_cycle == 0:
            sentence += f", once every {every // per_cycle} cycle{'s' if every // per_cycle != 1 else ''}"
        else:
            sentence += (f", {per_cycle / every:.3g} a cycle of {per_cycle} steps, so at a different point "
                         f"of each; adapting the saving to the time step would keep them on the cycle")
    return sentence + f"; restart files every {saving.restart_every} steps."
