"""Every setting a rigid-wall CFD `solver.xml` is written with, its default, and how it is read.

**One table, and the defaults in it are the only ones.** Every value the writer puts in a
`solver.xml` other than the mesh and the boundary conditions is a row here: what the panel draws
a field for, what it starts that field at, and what turns what was typed into what is written.
There is no second default anywhere -- not in the panel, not in the writer, not left to
svMultiPhysics -- so changing a default is changing one line here, and a `solver.xml` says every
value it was run with rather than leaving some to whichever svMultiPhysics read it.

**Where the defaults come from.** The linear solver's, the time stepping's and the saving's are
the lab's for rigid-wall CFD: 10000 steps of 1 ms, saving results and restarts every 100 steps
from the first. The rest are, for now, the Fontan workflow's `solver_template.xml`
(ClinicalWorkflows), less its scalar-transport equation: a template that has run every Fontan
case, standing in until the defaults meant for this panel replace it.

**Why every value goes through `parse`.** svMultiPhysics reads a number with `istringstream`,
which takes the longest numeric prefix and drops the rest without a word: `1,5e-4` is read as
`1`, `0.04P` as `0.04`. So nothing typed is written as typed. It is parsed here into a number
of the setting's kind, refused if it is not one, and written from the number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

BOOL, INT, FLOAT, TEXT = "bool", "int", "float", "text"


class SettingError(ValueError):
    """Raised when a value cannot be what its setting is."""


@dataclass(frozen=True)
class Setting:
    key: str
    """Stable, and what a value is saved under: the panel's scene keeps `{key: value}`."""
    section: str
    label: str
    kind: str
    default: Any
    note: str = ""
    choices: tuple = ()
    minimum: float | None = None

    def parse(self, value) -> Any:
        """`value` -- typed, saved, or already of the right kind -- as this setting's kind."""
        if self.kind == BOOL:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes", "on"):
                return True
            if text in ("false", "0", "no", "off"):
                return False
            raise SettingError(f"{self.label}: {value!r} is not true or false.")
        if self.kind == TEXT:
            text = str(value).strip()
            if not text or any(character.isspace() for character in text):
                raise SettingError(f"{self.label} has to be one word, not {value!r}.")
            if self.choices and text not in self.choices:
                raise SettingError(f"{self.label} has to be one of {', '.join(self.choices)}, "
                                   f"not {text!r}.")
            return text
        try:
            number = int(str(value).strip()) if self.kind == INT else float(str(value).strip())
        except ValueError:
            raise SettingError(f"{self.label}: {value!r} is not a"
                               f"{'n integer' if self.kind == INT else ' number'}.") from None
        if not math.isfinite(number):
            raise SettingError(f"{self.label} has to be a finite number.")
        if self.minimum is not None and number < self.minimum:
            raise SettingError(f"{self.label} has to be at least {self.minimum:g}, not {number:g}.")
        return number

    def format(self, value) -> str:
        """What is written into the `solver.xml` for `value`, after `parse`."""
        value = self.parse(value)
        if self.kind == BOOL:
            return "true" if value else "false"
        if self.kind == FLOAT:
            # The shortest text that reads back as the same number: what was typed, give or
            # take how it was spelt.
            return repr(float(value))
        return str(value)


GENERAL = "General"
MESH = "Mesh"
FLUID = "Fluid"
LINEAR_SOLVER = "Linear solver"
BOUNDARY_CONDITIONS = "Boundary conditions"
OUTPUTS = "Outputs"
SECTIONS = (GENERAL, MESH, FLUID, LINEAR_SOLVER, BOUNDARY_CONDITIONS, OUTPUTS)

# What a section says above its fields, where something about it is fixed rather than set: the
# fields are only what can be changed, and a section that leaves part of itself unsaid reads as
# though that part were up to the solver.
SECTION_NOTES = {
    LINEAR_SOLVER: 'For a CFD simulation, default LS type is "NS" and Linear_algebra type is '
                   '"fsils".',
}

# The linear solver a CFD simulation is written with, fixed rather than set: the Navier-Stokes
# solver, through svMultiPhysics' own linear algebra, which is what every rigid-wall case here
# runs and the one that needs nothing beyond svMultiPhysics built. Trilinos or PETSc would have to
# be built into the solver, and the panel is for a class running the solver as it comes.
LINEAR_SOLVER_TYPE = "NS"
LINEAR_ALGEBRA_TYPE = "fsils"
PRECONDITIONER = "fsils"

SETTINGS = (
    # GeneralSimulationParameters, as the Fontan template sets them.
    Setting("general.continue_previous_simulation", GENERAL, "Continue previous simulation", BOOL,
            False, "Start from the last restart file in the run folder rather than from rest."),
    Setting("general.number_of_spatial_dimensions", GENERAL, "Spatial dimensions", INT, 3,
            "3 for a vascular model.", minimum=2),
    Setting("general.number_of_time_steps", GENERAL, "Number of time steps", INT, 10000,
            "10 s of 1 ms steps by default: a dozen cycles at a resting heart rate. For a whole "
            "number of cycles, make it cycles times steps per cycle.", minimum=1),
    Setting("general.time_step_size", GENERAL, "Time step size", FLOAT, 1e-3,
            "Seconds.", minimum=0.0),
    Setting("general.spectral_radius_of_infinite_time_step", GENERAL,
            "Spectral radius of infinite time step", FLOAT, 0.5,
            "The generalised-alpha integrator's damping of high frequencies, from 0 (most) to "
            "1 (none).", minimum=0.0),
    Setting("general.searched_file_name_to_trigger_stop", GENERAL, "Stop file name", TEXT,
            "STOP_SIM", "A file of this name in the run folder stops the run cleanly."),
    Setting("general.save_results_to_vtk_format", GENERAL, "Save results to VTK", BOOL, True),
    Setting("general.name_prefix_of_saved_vtk_files", GENERAL, "Result file prefix", TEXT,
            "result"),
    Setting("general.increment_in_saving_vtk_files", GENERAL, "Save results every N steps", INT,
            100, minimum=1),
    Setting("general.start_saving_after_time_step", GENERAL, "Start saving after step", INT,
            1, "Saving from the first step keeps the whole run, transient and all; a later step "
            "keeps only the cycles after it.", minimum=0),
    Setting("general.increment_in_saving_restart_files", GENERAL, "Save restart every N steps",
            INT, 100, "How far back a run restarts from if it stops.", minimum=1),
    Setting("general.convert_bin_to_vtk_format", GENERAL, "Convert BIN to VTK", BOOL, False),
    Setting("general.verbose", GENERAL, "Verbose", BOOL, True),
    Setting("general.warning", GENERAL, "Warnings", BOOL, False),
    Setting("general.debug", GENERAL, "Debug", BOOL, False),

    # Add_mesh.
    Setting("mesh.scale_factor", MESH, "Mesh scale factor", FLOAT, 0.1,
            "What the mesh's coordinates are multiplied by. Slicer's are millimetres and the "
            "solver works in centimetres, so 0.1. Always written: left out, the solver takes "
            "1.0, and a mesh in mm becomes a domain ten times too large.", minimum=0.0),

    # Add_equation type="fluid".
    Setting("fluid.coupled", FLUID, "Coupled", BOOL, True),
    Setting("fluid.min_iterations", FLUID, "Min iterations", INT, 2, minimum=1),
    Setting("fluid.max_iterations", FLUID, "Max iterations", INT, 5, minimum=1),
    Setting("fluid.tolerance", FLUID, "Tolerance", FLOAT, 1e-3, minimum=0.0),
    Setting("fluid.backflow_stabilization_coefficient", FLUID, "Backflow stabilization", FLOAT,
            0.2, "Damps flow re-entering through an outlet, which otherwise grows and stops "
            "the run.", minimum=0.0),
    Setting("fluid.density", FLUID, "Density", FLOAT, 1.06, "g/cm³.", minimum=0.0),
    Setting("fluid.viscosity", FLUID, "Viscosity", FLOAT, 0.04,
            "Poise, g/(cm·s); a Newtonian fluid.", minimum=0.0),

    # The fluid's LS type="NS" with fsils (fixed; see LINEAR_SOLVER_TYPE): the lab's rigid-wall
    # defaults, exactly. Absolute_tolerance is not among them and is left to svMultiPhysics' own
    # (1e-10), which the Fontan template tightens to 1e-17; that was considered and not taken.
    Setting("ls.max_iterations", LINEAR_SOLVER, "Max iterations", INT, 10, minimum=1),
    Setting("ls.ns_gm_max_iterations", LINEAR_SOLVER, "NS GM max iterations", INT, 200, minimum=1),
    Setting("ls.ns_cg_max_iterations", LINEAR_SOLVER, "NS CG max iterations", INT, 500, minimum=1),
    Setting("ls.tolerance", LINEAR_SOLVER, "Tolerance", FLOAT, 0.4, minimum=0.0),
    Setting("ls.ns_gm_tolerance", LINEAR_SOLVER, "NS GM tolerance", FLOAT, 0.01, minimum=0.0),
    Setting("ls.ns_cg_tolerance", LINEAR_SOLVER, "NS CG tolerance", FLOAT, 0.2, minimum=0.0),
    Setting("ls.krylov_space_dimension", LINEAR_SOLVER, "Krylov space dimension", INT, 50,
            minimum=1),

    # How the caps' conditions are written, where the 0D conditions they are shared with do not
    # say: a 0D model has no velocity profile and no initial pressure.
    Setting("bc.inflow_profile", BOUNDARY_CONDITIONS, "Inflow profile", TEXT, "Parabolic",
            "The velocity profile an inflow is imposed with across its cap.",
            choices=("Parabolic", "Flat")),
    Setting("bc.impose_flux", BOUNDARY_CONDITIONS, "Impose flux", BOOL, True,
            "Scale the profile so that the flow through the cap is the inflow's, whatever the "
            "cap's shape. Off, the values are velocities rather than flows."),
    Setting("bc.fourier_coefficients", BOUNDARY_CONDITIONS, "Inflow Fourier coefficients", INT,
            16, "How many Fourier coefficients the solver fits an inflow waveform with -- the "
            "second number of a .flow file's header, not its column count. Two would smooth a "
            "waveform into very nearly a sinusoid.", minimum=1),
    Setting("bc.rcr_initial_pressure", BOUNDARY_CONDITIONS, "RCR initial pressure", FLOAT, 0.0,
            "dyn/cm². What every RCR outlet starts at. 0 starts them empty, which costs the run "
            "a few cycles of filling."),

    # Output: what the solver writes. The Fontan template's selection.
    Setting("output.velocity", OUTPUTS, "Velocity", BOOL, True),
    Setting("output.pressure", OUTPUTS, "Pressure", BOOL, True),
    Setting("output.traction", OUTPUTS, "Traction", BOOL, True),
    Setting("output.vorticity", OUTPUTS, "Vorticity", BOOL, True),
    Setting("output.divergence", OUTPUTS, "Divergence", BOOL, True),
    Setting("output.wss", OUTPUTS, "Wall shear stress", BOOL, True),
    Setting("output.boundary_pressure", OUTPUTS, "Pressure per face (B_INT)", BOOL, True,
            "The pressure integrated over each face, per step, which is what a pressure at a "
            "cap is read from."),
    Setting("output.boundary_velocity", OUTPUTS, "Flow per face (B_INT)", BOOL, True,
            "The velocity integrated over each face -- the flow through it -- per step."),
    Setting("output.volume_pressure", OUTPUTS, "Pressure over the volume (V_INT)", BOOL, True),
)

BY_KEY = {setting.key: setting for setting in SETTINGS}


def defaults() -> dict:
    return {setting.key: setting.default for setting in SETTINGS}


def resolve(values: Mapping[str, Any] | None = None) -> dict:
    """Every setting, parsed: what was given where it was given, the default elsewhere.

    A key that is no setting is refused rather than ignored, because a misspelt one would
    otherwise leave the default in force with nothing said.
    """
    values = dict(values or {})
    unknown = sorted(set(values) - set(BY_KEY))
    if unknown:
        raise SettingError(f"No such setting: {', '.join(unknown)}.")
    resolved = {}
    for setting in SETTINGS:
        resolved[setting.key] = setting.parse(values.get(setting.key, setting.default))
    if resolved["fluid.min_iterations"] > resolved["fluid.max_iterations"]:
        raise SettingError("The fluid's min iterations are more than its max iterations.")
    return resolved


def changed(values: Mapping[str, Any]) -> dict:
    """Only the values that differ from their default: what a scene needs to keep."""
    resolved = resolve(values)
    return {key: value for key, value in resolved.items() if value != BY_KEY[key].default}
