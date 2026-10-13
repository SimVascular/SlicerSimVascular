"""What a 0D solution says at each cap, and along the centerlines it was built on.

svZeroDSolver reports per *vessel*: the flow and pressure at both ends of every segment of
every branch. What anyone asks of a result is per *face* -- what is the pressure at the RPA,
how much of the venous return goes left -- and where on the anatomy, so this reads the solver
input beside the results to say which vessel end each cap is, and maps the vessels back onto
the centerlines they were cut from.

The sign of a flow is turned to mean the same thing at every cap: positive in the direction its
condition drives it, into the model at an inflow and out of it at an outlet. svZeroDSolver
counts flow down each vessel, so at an inflow prescribed on a vessel's outlet end -- every
inflow but the inlet's, since the centerlines run from the inlet -- its own number is negative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svromsetup.boundary_conditions import MMHG

PRESSURE_ARRAY_NAME = "Pressure [mmHg]"
FLOW_ARRAY_NAME = "Flow [mL/s]"

# How the package names a vessel: its centerline branch and its segment along that branch.
_VESSEL_NAME = re.compile(r"^branch(\d+)_seg(\d+)$")


@dataclass(frozen=True)
class FaceResult:
    name: str
    bc_name: str
    bc_type: str
    vessel: str
    end: str
    """Which end of the vessel the cap is: "inlet" or "outlet"."""
    time: np.ndarray
    flow: np.ndarray
    """mL/s, positive into the model at an inflow and out of it anywhere else."""
    pressure: np.ndarray
    """dyn/cm²."""

    @property
    def is_inflow(self) -> bool:
        return self.bc_type == "FLOW"

    @property
    def mean_flow(self) -> float:
        return _cycle_mean(self.time, self.flow)

    @property
    def mean_pressure_mmhg(self) -> float:
        return _cycle_mean(self.time, self.pressure) / MMHG

    @property
    def pressure_range_mmhg(self):
        return float(self.pressure.min() / MMHG), float(self.pressure.max() / MMHG)


def face_results(config, results, inlet_name: str) -> list:
    """One FaceResult per cap, in the order the solver input lists its conditions.

    :param config: the solver input, as a dict or a path to it.
    :param inlet_name: the inlet's face, which the package's condition does not name -- it is
      called `INFLOW` -- and which only the caller knows.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    ends = {}
    for vessel in config.get("vessels", []):
        for end, bc_name in vessel.get("boundary_conditions", {}).items():
            ends[bc_name] = (vessel["vessel_name"], end)

    found = []
    for bc in config.get("boundary_conditions", []):
        bc_name, bc_type = bc["bc_name"], bc["bc_type"]
        if bc_name not in ends:
            continue
        vessel, end = ends[bc_name]
        result = results.get(vessel)
        if result is None:
            continue
        if end == "inlet":
            flow, pressure = result.flow_in, result.pressure_in
        else:
            flow, pressure = result.flow_out, result.pressure_out
            if bc_type == "FLOW":
                flow = -flow
        found.append(FaceResult(cap_name(bc_name, bc_type, inlet_name), bc_name, bc_type, vessel,
                                end, result.time, flow, pressure))
    return found


def cap_name(bc_name: str, bc_type: str, inlet_name: str) -> str:
    """The face a condition in the solver input is on.

    The package names a condition after its kind and its face, `RCR_cap_lpa_a`, except the
    inlet's, which it calls `INFLOW` whatever the face is -- so that one is the caller's to say.
    """
    if bc_name == "INFLOW":
        return inlet_name
    prefix = f"{bc_type}_"
    return bc_name[len(prefix):] if bc_name.startswith(prefix) else bc_name


def write_face_results_csv(path, face_results) -> Path:
    """Every cap's flow and pressure over the last cycle, a column each, into one CSV file.

    One row per time point, with the time first and then each cap's flow and pressure, in mL/s
    and mmHg, under the cap's name: the per-face view of what svZeroDSolver's own CSV holds per
    vessel segment, which is the one to read results against the faces by. Flow keeps the sign
    convention of `FaceResult`, so an inflow and an outlet both read positive.
    """
    if not face_results:
        raise ValueError("There are no results to write.")
    time = np.asarray(face_results[0].time, dtype=float)
    columns = [("time [s]", time)]
    for face in face_results:
        if len(face.time) != len(time):
            raise ValueError(f"{face.name} has {len(face.time)} time points and the others "
                             f"{len(time)}; they come from one solve and should not differ.")
        columns.append((f"{face.name} flow [mL/s]", np.asarray(face.flow, dtype=float)))
        columns.append((f"{face.name} pressure [mmHg]", np.asarray(face.pressure, dtype=float) / MMHG))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(",".join(name for name, _values in columns) + "\n")
        for row in np.column_stack([values for _name, values in columns]):
            handle.write(",".join(f"{value:.10g}" for value in row) + "\n")
    return path


def centerline_results(centerline, config, results):
    """A copy of the centerlines with the cycle-averaged pressure and flow at every point.

    Each branch is the package's chain of vessels `branch<b>_seg<s>`, laid end to end along the
    branch in proportion to their lengths, so a point's place along its branch's `Path` says which
    segment it is in. Pressure is interpolated between the segment's two ends, which is what the
    model says about the inside of a vessel; flow is the segment's, which is the same all along it
    but for the little its compliance stores. Points inside a bifurcation belong to no branch, and
    take the value of the nearest point that does.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    copy = vtk.vtkPolyData()
    copy.DeepCopy(centerline)
    point_data = copy.GetPointData()

    names, fractions = _segment_positions(copy, config, only=set(results))
    pressure = np.full(len(names), np.nan)
    flow = np.full(len(names), np.nan)
    for name in set(names) - {""}:
        result = results[name]
        here = names == name
        p_in = _cycle_mean(result.time, result.pressure_in)
        p_out = _cycle_mean(result.time, result.pressure_out)
        pressure[here] = (p_in + (p_out - p_in) * fractions[here]) / MMHG
        flow[here] = (_cycle_mean(result.time, result.flow_in)
                      + _cycle_mean(result.time, result.flow_out)) / 2.0

    known = np.flatnonzero(np.isfinite(pressure))
    unknown = np.flatnonzero(~np.isfinite(pressure))
    if len(known) and len(unknown):
        coordinates = vtk_to_numpy(copy.GetPoints().GetData()).astype(float)
        for point in unknown:
            nearest = known[np.argmin(np.linalg.norm(coordinates[known] - coordinates[point], axis=1))]
            pressure[point], flow[point] = pressure[nearest], flow[nearest]

    for name, values in ((PRESSURE_ARRAY_NAME, pressure), (FLOW_ARRAY_NAME, flow)):
        array = numpy_to_vtk(np.nan_to_num(values), deep=True)
        array.SetName(name)
        point_data.AddArray(array)
    return copy


def vessel_lines(centerline, config, vessel_name: str):
    """The stretch of the centerlines one vessel was cut from, as lines: which vessel it is, shown.

    By the same rule as `centerline_results`, so that a vessel picked out on the anatomy is the
    stretch its results were coloured on. Lines rather than points, one per run of the vessel's
    points along each centerline, because a branch is traced once per centerline through it and
    its points come in parallel runs: joined in one polyline, they would zigzag between them. The
    inscribed sphere radius goes with them, for drawing the stretch as the vessel it is.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    names, _fractions = _segment_positions(centerline, config)
    inside = names == vessel_name
    radius = centerline.GetPointData().GetArray("MaximumInscribedSphereRadius")
    points, lines, radii, kept = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkDoubleArray(), {}
    radii.SetName("MaximumInscribedSphereRadius")

    def add(run):
        lines.InsertNextCell(len(run))
        for point in run:
            if point not in kept:
                kept[point] = points.InsertNextPoint(centerline.GetPoint(point))
                radii.InsertNextValue(radius.GetValue(point) if radius is not None else 0.0)
            lines.InsertCellPoint(kept[point])

    for cell in range(centerline.GetNumberOfCells()):
        ids = centerline.GetCell(cell).GetPointIds()
        run = []
        for index in range(ids.GetNumberOfIds()):
            point = ids.GetId(index)
            if inside[point]:
                run.append(point)
                continue
            if len(run) >= 2:
                add(run)
            run = []
        if len(run) >= 2:
            add(run)
    lines_only = vtk.vtkPolyData()
    lines_only.SetPoints(points)
    lines_only.SetLines(lines)
    lines_only.GetPointData().AddArray(radii)
    return lines_only


def _segment_positions(centerline, config, only=None):
    """Which vessel each centerline point is in, and how far along it: (names, fractions).

    A branch's vessels are laid end to end along it in proportion to their lengths, so a point's
    place along its branch's `Path` says which it is in. A point inside a bifurcation is in no
    branch, and gets the name "". `only` limits it to the vessels named, which the others' points
    then share in proportion, as when a vessel has no results.
    """
    point_data = centerline.GetPointData()
    branch_ids = vtk_to_numpy(point_data.GetArray("BranchId")).astype(np.int64)
    path = vtk_to_numpy(point_data.GetArray("Path")).astype(float)
    segments = {}
    for vessel in config.get("vessels", []):
        name = vessel["vessel_name"]
        match = _VESSEL_NAME.match(name)
        if match and (only is None or name in only):
            segments.setdefault(int(match.group(1)), []).append(
                (int(match.group(2)), float(vessel["vessel_length"]), name))

    names = np.full(len(branch_ids), "", dtype=object)
    fractions = np.zeros(len(branch_ids))
    for branch, pieces in segments.items():
        points = np.flatnonzero(branch_ids == branch)
        if not len(points):
            continue
        pieces.sort(key=lambda piece: piece[0])
        lengths = np.array([length for _index, length, _name in pieces])
        bounds = np.concatenate([[0.0], np.cumsum(lengths)]) / lengths.sum()
        along = path[points] - path[points].min()
        along = along / along.max() if along.max() > 0 else np.zeros_like(along)
        which = np.clip(np.searchsorted(bounds, along, side="right") - 1, 0, len(pieces) - 1)
        for index, (_segment, _length, name) in enumerate(pieces):
            here = which == index
            span = bounds[index + 1] - bounds[index]
            names[points[here]] = name
            fractions[points[here]] = (along[here] - bounds[index]) / span if span > 0 else 0.0
    return names, fractions


def _cycle_mean(time, values) -> float:
    """The time-average over what the solver wrote, which is the last cycle by default."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(time) < 2 or time[-1] == time[0]:
        return float(values.mean())
    return float(_trapezoid(values, time) / (time[-1] - time[0]))


# numpy 2 renamed trapz, and Slicer's numpy and a terminal's need not be the same major version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
