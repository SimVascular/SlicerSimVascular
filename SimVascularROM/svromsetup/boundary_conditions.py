"""What happens at each cap: a flow coming in, or a resistance (with or without a compliance) out.

A reduced-order model is a network of vessels, and the network is closed by one condition per
cap. Three kinds cover what a vascular model takes, and each is what svZeroDSolver calls it:

- **Inflow**, a prescribed flow entering the model -- a waveform, or a steady flow, which is a
  waveform that does not change. Positive is *into* the model. A Fontan has several: the venae
  cavae, the hepatic veins and the azygous all enter, and the pulmonary arteries all leave.
- **RCR**, a proximal resistance, a compliance and a distal resistance, draining to a distal
  pressure: the Windkessel that stands in for the vascular bed beyond an outlet.
- **Resistance**, a resistance alone, draining to zero pressure: what an outlet into a
  low-impedance bed -- the lungs, in a Fontan -- is usually given. It has no distal pressure,
  because svMultiPhysics' Resistance condition has none (it is `P = R Q`), and the same
  conditions are written for the 0D and the 3D solver: an outlet that needs a pressure to drain
  to is an RCR.

Units are cgs throughout, as both solvers take them: flow in mL/s, resistance in dyn·s/cm⁵,
compliance in cm⁵/dyn, pressure in dyn/cm². Nothing here converts; the panel says so beside
the table, which is where a value in mmHg would be typed.

Every condition is kept per face *name*, not per face id, by the files the solvers read: those
are SimVascular's `rcrt.dat` and `resistance.dat`, both keyed by face name, and the reader here
takes them from a SimVascular project as they are.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# The kinds, as the panel lists them and as they are saved.
INFLOW = "Inflow"
RCR_KIND = "RCR"
RESISTANCE_KIND = "Resistance"
KINDS = (INFLOW, RCR_KIND, RESISTANCE_KIND)

# What one mmHg is in dyn/cm², for the panel to report pressures in the unit they are read in.
MMHG = 1333.22

_SEPARATORS = re.compile(r"[,;\s]+")


class BoundaryConditionError(ValueError):
    """Raised when a condition cannot be read, or the set of them cannot close the model."""


@dataclass(frozen=True)
class Inflow:
    """A flow entering the model at a cap, over one cardiac cycle.

    The cycle is the waveform's own last time: both solvers repeat a waveform over it, so a
    steady flow is two points, at 0 and at the period, carrying the same value.
    """

    time: tuple
    flow: tuple
    source: str = field(default="", compare=False)
    """The file the waveform was read from, if it was one: what the table shows for it, so that
    the condition on a cap says which file it is. Not compared -- the same numbers are the same
    inflow wherever they came from -- and not needed to run it, since the numbers are kept with
    the scene: a file moved or deleted since leaves the condition as it was."""
    kind = INFLOW

    def __post_init__(self):
        if len(self.time) != len(self.flow):
            raise BoundaryConditionError("An inflow needs as many flows as times.")
        if len(self.time) < 2:
            raise BoundaryConditionError(
                "An inflow needs two points at least: the waveform is repeated over its last "
                "time, so one point has no period to repeat over.")
        if any(later <= earlier for earlier, later in zip(self.time, self.time[1:])):
            raise BoundaryConditionError("An inflow's times have to increase.")
        if self.time[0] != 0.0:
            raise BoundaryConditionError(
                "An inflow has to start at time 0: the cycle is taken as running from 0 to "
                "its last time.")

    @classmethod
    def steady(cls, flow: float, period: float = 1.0) -> "Inflow":
        return cls((0.0, float(period)), (float(flow), float(flow)))

    @property
    def period(self) -> float:
        return float(self.time[-1])

    @property
    def is_steady(self) -> bool:
        return len(set(self.flow)) == 1

    @property
    def mean(self) -> float:
        """The cycle-averaged flow, by the trapezoidal rule over the waveform's own points."""
        total = sum((t1 - t0) * (q0 + q1) / 2.0 for t0, t1, q0, q1 in
                    zip(self.time, self.time[1:], self.flow, self.flow[1:]))
        return total / self.period

    def flipped(self) -> "Inflow":
        return Inflow(self.time, tuple(-q for q in self.flow), self.source)


@dataclass(frozen=True)
class RCR:
    Rp: float
    C: float
    Rd: float
    Pd: float = 0.0
    kind = RCR_KIND


@dataclass(frozen=True)
class Resistance:
    R: float
    kind = RESISTANCE_KIND


# -- as typed in the table ----------------------------------------------------
# What each kind's values are, in the order they are typed. The panel shows these as the cell's
# hint, so the order is the one thing to get right and is said once, here.
VALUE_ORDER = {
    INFLOW: ("Q",),
    RCR_KIND: ("Rp", "C", "Rd", "Pd"),
    RESISTANCE_KIND: ("R",),
}


def to_text(condition) -> str:
    """The condition as the table shows it, which is also what parses back to it.

    A waveform is the exception: it is loaded from a file rather than typed, so it is shown as
    the file it came from, and typing a number over that replaces it with a steady flow. Only a
    waveform that came from nowhere -- made in code, or saved before its file was kept -- is
    described instead.
    """
    if condition is None:
        return ""
    if isinstance(condition, Inflow):
        if condition.source:
            return condition.source
        if condition.is_steady:
            return _number(condition.flow[0])
        return (f"waveform, {len(condition.time)} points over {_number(condition.period)} s, "
                f"mean {_number(condition.mean)}")
    if isinstance(condition, RCR):
        return ", ".join(_number(value) for value in (condition.Rp, condition.C, condition.Rd,
                                                      condition.Pd))
    if isinstance(condition, Resistance):
        return _number(condition.R)
    raise BoundaryConditionError(f"Not a boundary condition: {condition!r}")


def from_text(kind: str, text: str, previous=None, period: float = 1.0):
    """A condition of `kind` out of what was typed, or None for an empty cell.

    `previous` is what the cell held before. It is what keeps a loaded waveform when its
    description is left as it was -- the table hands back the cell's text whether or not it was
    edited -- and `period` is what a steady inflow is spread over, so that it can sit beside
    waveforms of the other inflows without its period disagreeing with theirs.
    """
    text = (text or "").strip()
    if not text:
        return None
    if previous is not None and getattr(previous, "kind", None) == kind and text == to_text(previous):
        return previous
    try:
        values = [float(value) for value in _SEPARATORS.split(text) if value]
    except ValueError:
        raise BoundaryConditionError(
            f"Could not read {text!r} as numbers. {kind} takes "
            f"{', '.join(VALUE_ORDER[kind])}, in that order.") from None
    names = VALUE_ORDER[kind]
    # Every value but an RCR's distal pressure has to be there; that one is 0 when left out.
    required = len(names) - 1 if kind == RCR_KIND else len(names)
    if not required <= len(values) <= len(names):
        raise BoundaryConditionError(
            f"{kind} takes {', '.join(names)}"
            + (f" ({names[-1]} may be left out, and is then 0)" if kind == RCR_KIND else "")
            + f"; {len(values)} value(s) were given.")
    if any(not math.isfinite(value) for value in values):
        raise BoundaryConditionError("Every value has to be a finite number.")
    if kind == INFLOW:
        return Inflow.steady(values[0], period)
    if kind == RCR_KIND:
        return RCR(*values)
    return Resistance(*values)


def _number(value: float) -> str:
    return f"{value:.6g}"


# -- kept on the mesh -----------------------------------------------------------
def to_json(conditions: Mapping[int, object]) -> str:
    """`{"<face id>": {...}}`, sorted, so a scene saved twice untouched is the same bytes twice."""
    stored = {}
    for face_id, condition in sorted(conditions.items()):
        if condition is None:
            continue
        if isinstance(condition, Inflow):
            entry = {"type": INFLOW, "t": list(condition.time), "Q": list(condition.flow)}
            if condition.source:
                entry["file"] = condition.source
        elif isinstance(condition, RCR):
            entry = {"type": RCR_KIND, "Rp": condition.Rp, "C": condition.C, "Rd": condition.Rd,
                     "Pd": condition.Pd}
        elif isinstance(condition, Resistance):
            entry = {"type": RESISTANCE_KIND, "R": condition.R}
        else:
            raise BoundaryConditionError(f"Not a boundary condition: {condition!r}")
        stored[str(face_id)] = entry
    return json.dumps(stored, separators=(",", ":"))


def from_json(text: str) -> dict:
    """The conditions back, by face id. An entry that cannot be read is dropped, not raised:
    a face left without a condition is something the panel shows, and a scene that will not
    open over one bad value is not."""
    if not text:
        return {}
    try:
        stored = json.loads(text)
    except ValueError:
        return {}
    conditions = {}
    for face_id, entry in stored.items():
        try:
            kind = entry["type"]
            if kind == INFLOW:
                condition = Inflow(tuple(float(t) for t in entry["t"]),
                                   tuple(float(q) for q in entry["Q"]), str(entry.get("file", "")))
            elif kind == RCR_KIND:
                condition = RCR(float(entry["Rp"]), float(entry["C"]), float(entry["Rd"]),
                                float(entry.get("Pd", 0.0)))
            elif kind == RESISTANCE_KIND:
                condition = Resistance(float(entry["R"]))
            else:
                continue
            conditions[int(face_id)] = condition
        except (KeyError, TypeError, ValueError):
            continue
    return conditions


# -- whether they close the model ------------------------------------------------
def problems(conditions: Mapping[int, object], cap_names: Mapping[int, str],
             inlet_face_id: int | None, *, source_required: bool = True) -> list:
    """What stops these conditions from closing the model, in words; empty when nothing does.

    Each is something the package or the solver would otherwise find later and say less
    clearly, or not say at all:

    - a cap with no condition, which the package refuses by name, after centerlines;
    - an inlet that is not an inflow -- centerlines start at the inlet, and the package
      prescribes its flow there, so anything else typed against it would be dropped;
    - no outlet at all. With every cap prescribing flow, nothing sets the pressure, and the
      solver's system is singular; it fails to converge rather than saying why;
    - inflows over different periods, which the package refuses: the cycle is the inlet's.

    `source_required` is False for a 3D model, which has no centerlines and so no source; the
    other three hold of it as they do of a 0D one.
    """
    found = []
    missing = [name for face_id, name in sorted(cap_names.items())
               if conditions.get(face_id) is None]
    if missing:
        found.append(f"{len(missing)} cap(s) have no boundary condition: {', '.join(missing)}.")
    if not source_required:
        pass
    elif inlet_face_id is None:
        found.append("Choose the source the centerlines start from.")
    elif inlet_face_id in cap_names and not isinstance(conditions.get(inlet_face_id), Inflow):
        found.append(f"The source, {cap_names[inlet_face_id]}, has to be an inflow: the centerlines "
                     "start there and its flow is what is prescribed on them.")
    given = {face_id: condition for face_id, condition in conditions.items()
             if face_id in cap_names and condition is not None}
    if given and not any(isinstance(condition, (RCR, Resistance)) for condition in given.values()):
        found.append("At least one cap has to be an RCR or a resistance: with flow prescribed "
                     "everywhere, nothing sets the pressure.")
    periods = sorted({condition.period for condition in given.values()
                      if isinstance(condition, Inflow) and not condition.is_steady})
    if len(periods) > 1:
        found.append("The inflow waveforms cover different periods ("
                     + ", ".join(f"{period:g} s" for period in periods)
                     + "); they have to cover the same cardiac cycle.")
    return found


def cycle_period(conditions) -> float:
    """The cardiac cycle: the period of the first waveform among them, or 1 s if all are steady.

    A steady inflow has no period of its own, so it takes this one. That is what lets a steady
    azygous flow sit beside a pulsatile hepatic one without the two disagreeing.
    """
    for condition in conditions:
        if isinstance(condition, Inflow) and not condition.is_steady:
            return condition.period
    return 1.0


def with_period(condition, period: float):
    """A steady inflow respread over `period`; anything else as it is."""
    if isinstance(condition, Inflow) and condition.is_steady and condition.period != period:
        return Inflow.steady(condition.flow[0], period)
    return condition


# -- SimVascular's files ---------------------------------------------------------
def read_flow_file(path) -> Inflow:
    """A waveform from a `.flow` file: time and flow per line, space- or comma-separated.

    As written, sign and all. A file made for svMultiPhysics carries an inflow as *negative* --
    its flow is measured along the outward normal -- and the reduced-order solvers take it as
    positive. Which one a file is cannot be read off the file, so the caller says so; see
    `as_inflow`.
    """
    times, flows = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        values = [value for value in _SEPARATORS.split(line.strip()) if value]
        if not values or values[0].startswith("#"):
            continue
        try:
            times.append(float(values[0]))
            flows.append(float(values[1]))
        except (IndexError, ValueError):
            raise BoundaryConditionError(
                f"{path}, line {number}: expected a time and a flow, found {line.strip()!r}."
            ) from None
    return Inflow(tuple(times), tuple(flows), str(Path(path).resolve()))


def as_inflow(waveform: Inflow):
    """The waveform as an inflow, positive in, and whether it had to be turned round to be one.

    A waveform whose mean is negative is taken to be one written for svMultiPhysics, where
    inflow is negative, rather than a flow out of the model: an outlet prescribing flow is
    not something this panel offers. Returned with a flag rather than flipped quietly, so that
    the panel can say it did.
    """
    if waveform.mean < 0:
        return waveform.flipped(), True
    return waveform, False


def write_flow_file(path, inflow: Inflow) -> Path:
    path = Path(path)
    path.write_text("".join(f"{t:.10g} {q:.10g}\n" for t, q in zip(inflow.time, inflow.flow)))
    return path


def read_rcrt(path) -> dict:
    """`{face name: RCR}` from a SimVascular `rcrt.dat`.

    The file opens with the number of distal pressure points, and each face's block repeats it:
    the face name, Rp, C, Rd, then that many time / distal pressure pairs. Only a constant
    distal pressure is read, because that is what the 0D writer takes; a varying one is refused
    by name rather than averaged.
    """
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return {}
    keyword = lines[0]
    found = {}
    index = 1
    while index < len(lines):
        if lines[index] != keyword:
            raise BoundaryConditionError(
                f"{path}: expected a block starting with {keyword!r}, found {lines[index]!r}.")
        try:
            name = lines[index + 1]
            rp, c, rd = (float(value) for value in lines[index + 2:index + 5])
            count = int(keyword)
            pressures = [float(lines[index + 5 + point].split()[1]) for point in range(count)]
        except (IndexError, ValueError):
            raise BoundaryConditionError(f"{path}: the block starting at line {index + 1} is "
                                         "not a face name, Rp, C, Rd and its pressures.") from None
        if len(set(pressures)) > 1:
            raise BoundaryConditionError(f"{path}: {name} has a distal pressure that varies in "
                                         "time, which the 0D writer cannot take.")
        found[name] = RCR(rp, c, rd, pressures[0] if pressures else 0.0)
        index += 5 + count
    return found


def write_rcrt(path, conditions: Mapping[str, RCR]) -> Path:
    """SimVascular's `rcrt.dat`, with the distal pressure held over the cycle 0 to 1.

    The pressure's times do not matter to the 0D writer, which takes the pressure and checks
    that both are the same; 0 and 1 are what SimVascular itself writes.
    """
    path = Path(path)
    blocks = ["2"]
    for name, rcr in conditions.items():
        blocks += ["2", name, f"{rcr.Rp:.10g}", f"{rcr.C:.10g}", f"{rcr.Rd:.10g}",
                   f"0.0 {rcr.Pd:.10g}", f"1.0 {rcr.Pd:.10g}"]
    path.write_text("\n".join(blocks) + "\n")
    return path


def read_resistance(path) -> dict:
    """`{face name: Resistance}` from a SimVascular `resistance.dat`: name, R, and a Pd of 0.

    SimVascular's file can carry a distal pressure as a third column, and a resistance here has
    none (see the module docstring). One of 0 is the same condition and is read; any other is
    refused by name rather than dropped, which would lower that outlet's pressure by it.
    """
    found = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        try:
            if len(values) not in (2, 3):
                raise ValueError
            numbers = [float(value) for value in values[1:]]
        except ValueError:
            raise BoundaryConditionError(f"{path}, line {number}: expected a face name and a "
                                         f"resistance, found {line.strip()!r}.") from None
        if len(numbers) == 2 and numbers[1] != 0.0:
            raise BoundaryConditionError(
                f"{path}: {values[0]} has a distal pressure of {numbers[1]:g}, and a resistance "
                "here has none. Give that outlet an RCR instead.")
        found[values[0]] = Resistance(numbers[0])
    return found


def write_resistance(path, conditions: Mapping[str, Resistance]) -> Path:
    path = Path(path)
    path.write_text("".join(f"{name} {resistance.R:.10g}\n"
                            for name, resistance in conditions.items()))
    return path
