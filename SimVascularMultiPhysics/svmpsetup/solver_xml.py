"""svMultiPhysics' `solver.xml` for a rigid-wall CFD case, built rather than templated.

Built element by element from `settings` and the caps' conditions, rather than written by filling
in a template, for the reason the Fontan workflow's writer refuses to ship one: what a template
carries besides its blanks is the most case-specific content in the file -- the equations, the
numerics, the outputs -- and a template applied to the wrong case converges on the wrong physics
with nothing said. Here every element is one of the settings' or one of the conditions', and a
setting is one row of one table.

Every face the mesh has is in the file. svMultiPhysics gives a boundary face that no `Add_BC`
names a zero-traction condition, which on a wall is a hole: a wall face left out of the file
would leak. So the walls are written from the face names as surely as the caps are, each with a
no-slip condition.

The sign of an inflow is turned round on the way in. The conditions shared with the 0D panel
take inflow as positive, into the model; svMultiPhysics measures a face's flow along its outward
normal, so flow into the domain is negative, in a `Value` and in a `.flow` file alike.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

from svmeshcomplete.face_table import Face
from svromsetup.boundary_conditions import RCR, Inflow, Resistance

from svmpsetup import settings as settings_module

ROOT_TAG = "svMultiPhysicsFile"
MESH_NAME = "msh"

# GeneralSimulationParameters, in the order svMultiPhysics' own test cases list them.
GENERAL_TAGS = (
    ("general.continue_previous_simulation", "Continue_previous_simulation"),
    ("general.number_of_spatial_dimensions", "Number_of_spatial_dimensions"),
    ("general.number_of_time_steps", "Number_of_time_steps"),
    ("general.time_step_size", "Time_step_size"),
    ("general.spectral_radius_of_infinite_time_step", "Spectral_radius_of_infinite_time_step"),
    ("general.searched_file_name_to_trigger_stop", "Searched_file_name_to_trigger_stop"),
    ("general.save_results_to_vtk_format", "Save_results_to_VTK_format"),
    ("general.name_prefix_of_saved_vtk_files", "Name_prefix_of_saved_VTK_files"),
    ("general.increment_in_saving_vtk_files", "Increment_in_saving_VTK_files"),
    ("general.start_saving_after_time_step", "Start_saving_after_time_step"),
    ("general.increment_in_saving_restart_files", "Increment_in_saving_restart_files"),
    ("general.convert_bin_to_vtk_format", "Convert_BIN_to_VTK_format"),
    ("general.verbose", "Verbose"),
    ("general.warning", "Warning"),
    ("general.debug", "Debug"),
)

FLUID_TAGS = (
    ("fluid.coupled", "Coupled"),
    ("fluid.min_iterations", "Min_iterations"),
    ("fluid.max_iterations", "Max_iterations"),
    ("fluid.tolerance", "Tolerance"),
    ("fluid.backflow_stabilization_coefficient", "Backflow_stabilization_coefficient"),
    ("fluid.density", "Density"),
)

LINEAR_SOLVER_TAGS = (
    ("ls.max_iterations", "Max_iterations"),
    ("ls.ns_gm_max_iterations", "NS_GM_max_iterations"),
    ("ls.ns_cg_max_iterations", "NS_CG_max_iterations"),
    ("ls.tolerance", "Tolerance"),
    ("ls.ns_gm_tolerance", "NS_GM_tolerance"),
    ("ls.ns_cg_tolerance", "NS_CG_tolerance"),
    ("ls.krylov_space_dimension", "Krylov_space_dimension"),
)

# Output groups: the type svMultiPhysics names the group by, and its quantities.
OUTPUT_GROUPS = (
    ("Spatial", (("output.velocity", "Velocity"), ("output.pressure", "Pressure"),
                 ("output.traction", "Traction"), ("output.vorticity", "Vorticity"),
                 ("output.divergence", "Divergence"), ("output.wss", "WSS"))),
    ("B_INT", (("output.boundary_pressure", "Pressure"), ("output.boundary_velocity", "Velocity"))),
    ("V_INT", (("output.volume_pressure", "Pressure"),)),
)


class SolverXmlError(ValueError):
    """Raised when the faces and conditions given cannot make a solver.xml."""


def solver_xml(faces: Sequence[Face], conditions_by_name: Mapping[str, object], values=None, *,
               volume_mesh_path: str, face_paths: Mapping[str, str],
               flow_file_paths: Mapping[str, str]) -> str:
    """The text of a `solver.xml` for these faces, these conditions and these settings.

    :param faces: every face of the mesh, caps and walls; see the module docstring for why every
      one.
    :param conditions_by_name: each cap's condition, by the cap's name.
    :param values: settings that differ from their defaults, by key; see `settings.resolve`.
    :param volume_mesh_path, face_paths, flow_file_paths: where each file is, as the solver is to
      read it -- relative to the folder the `solver.xml` is in, which is the one a job runs from.
    """
    resolved = settings_module.resolve(values)
    caps = [face for face in faces if face.is_cap]
    walls = [face for face in faces if face.is_wall]
    missing = [face.name for face in caps if face.name not in conditions_by_name]
    if missing:
        raise SolverXmlError(f"{len(missing)} cap(s) have no boundary condition: "
                             f"{', '.join(missing)}.")
    if not walls:
        raise SolverXmlError("The mesh has no wall face, so nothing would be given a no-slip "
                             "condition.")

    root = ET.Element(ROOT_TAG, {"version": "0.1"})
    general = ET.SubElement(root, "GeneralSimulationParameters")
    for key, tag in GENERAL_TAGS:
        _setting(general, tag, key, resolved)

    mesh = ET.SubElement(root, "Add_mesh", {"name": MESH_NAME})
    ET.SubElement(mesh, "Mesh_file_path").text = volume_mesh_path
    for face in faces:
        if face.name not in face_paths:
            raise SolverXmlError(f"No file for the face {face.name}.")
        added = ET.SubElement(mesh, "Add_face", {"name": face.name})
        ET.SubElement(added, "Face_file_path").text = face_paths[face.name]
    _setting(mesh, "Mesh_scale_factor", "mesh.scale_factor", resolved)

    fluid = ET.SubElement(root, "Add_equation", {"type": "fluid"})
    for key, tag in FLUID_TAGS:
        _setting(fluid, tag, key, resolved)
    viscosity = ET.SubElement(fluid, "Viscosity", {"model": "Constant"})
    _setting(viscosity, "Value", "fluid.viscosity", resolved)

    for group, quantities in OUTPUT_GROUPS:
        wanted = [(key, tag) for key, tag in quantities if resolved[key]]
        if not wanted:
            continue
        output = ET.SubElement(fluid, "Output", {"type": group})
        for key, tag in wanted:
            _setting(output, tag, key, resolved)

    linear_solver = ET.SubElement(fluid, "LS", {"type": settings_module.LINEAR_SOLVER_TYPE})
    algebra = ET.SubElement(linear_solver, "Linear_algebra",
                            {"type": settings_module.LINEAR_ALGEBRA_TYPE})
    ET.SubElement(algebra, "Preconditioner").text = settings_module.PRECONDITIONER
    for key, tag in LINEAR_SOLVER_TAGS:
        _setting(linear_solver, tag, key, resolved)

    for face in caps:
        fluid.append(_cap_condition(face.name, conditions_by_name[face.name], resolved,
                                    flow_file_paths))
    for face in walls:
        bc = ET.SubElement(fluid, "Add_BC", {"name": face.name})
        ET.SubElement(bc, "Type").text = "Dir"
        ET.SubElement(bc, "Time_dependence").text = "Steady"
        ET.SubElement(bc, "Value").text = "0.0"

    ET.indent(root, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8" ?>\n'
            "<!-- Written by SimVascular MultiPhysics (SlicerSimVascular): rigid-wall CFD. -->\n"
            + ET.tostring(root, encoding="unicode") + "\n")


def _cap_condition(name, condition, resolved, flow_file_paths) -> ET.Element:
    bc = ET.Element("Add_BC", {"name": name})
    if isinstance(condition, Inflow):
        ET.SubElement(bc, "Type").text = "Dir"
        if condition.is_steady and not condition.source:
            ET.SubElement(bc, "Time_dependence").text = "Steady"
            # Negative: into the domain is against the face's outward normal.
            ET.SubElement(bc, "Value").text = repr(-float(condition.flow[0]))
        else:
            if name not in flow_file_paths:
                raise SolverXmlError(f"No flow file for {name}'s inflow waveform.")
            ET.SubElement(bc, "Time_dependence").text = "Unsteady"
            ET.SubElement(bc, "Temporal_values_file_path").text = flow_file_paths[name]
        _setting(bc, "Profile", "bc.inflow_profile", resolved)
        _setting(bc, "Impose_flux", "bc.impose_flux", resolved)
    elif isinstance(condition, RCR):
        ET.SubElement(bc, "Type").text = "Neu"
        ET.SubElement(bc, "Time_dependence").text = "RCR"
        values = ET.SubElement(bc, "RCR_values")
        ET.SubElement(values, "Capacitance").text = repr(float(condition.C))
        ET.SubElement(values, "Distal_resistance").text = repr(float(condition.Rd))
        ET.SubElement(values, "Proximal_resistance").text = repr(float(condition.Rp))
        ET.SubElement(values, "Distal_pressure").text = repr(float(condition.Pd))
        _setting(values, "Initial_pressure", "bc.rcr_initial_pressure", resolved)
    elif isinstance(condition, Resistance):
        ET.SubElement(bc, "Type").text = "Neu"
        ET.SubElement(bc, "Time_dependence").text = "Resistance"
        ET.SubElement(bc, "Value").text = repr(float(condition.R))
    else:
        raise SolverXmlError(f"{name} has a condition the 3D writer does not know: {condition!r}.")
    return bc


def _setting(parent, tag, key, resolved) -> None:
    ET.SubElement(parent, tag).text = settings_module.BY_KEY[key].format(resolved[key])


def write_flow_file(path, inflow: Inflow, coefficients: int) -> None:
    """A waveform as svMultiPhysics reads one: a header, then time and flow, inflow negative.

    The header is the number of time points and the number of Fourier coefficients the solver
    fits the waveform with -- not the number of columns, which is the mistake the format invites
    and which smooths a waveform into very nearly a sinusoid without a word. It is written from
    the series, so that it cannot disagree with the lines under it: the solver reads that many
    lines and stops early or runs out otherwise.
    """
    lines = [f"{len(inflow.time)} {int(coefficients)}"]
    lines += [f"{float(t)!r} {-float(q)!r}" for t, q in zip(inflow.time, inflow.flow)]
    with open(path, "w", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
