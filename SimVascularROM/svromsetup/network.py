"""The 0D model as the network it is: what is joined to what, and what each block holds.

svZeroDSolver's input is three lists -- vessels, junctions, boundary conditions -- joined by name
and number, and its results are a row per vessel per time point. Neither shows the model. This
reads the input as a directed graph, a node for every condition, vessel segment and junction and
an edge for every connection, and lays it out in layers from the source, for a panel to draw.

The layout is worked out here rather than by Graphviz. Graphviz is a program, not a Python
package, so pip cannot install it, and a machine without it would draw nothing; and a network
traced along centerlines is a tree, which a layered layout draws as clearly as anything general
would.

Edges run the way the model was traced, from the source outwards, which is not always the way
the blood goes: an inflow prescribed at a centerline's far end -- a Fontan's hepatic veins -- is
an edge out of the model with the flow coming in, which is what the sign of its condition says
(see `results`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svromsetup import boundary_conditions as bcs
from svromsetup.results import _cycle_mean, cap_name

VESSEL = "vessel"
JUNCTION = "junction"
BOUNDARY = "boundary"

# How the package names a vessel, and what a block in a crowded graph has room for of it.
_VESSEL_NAME = re.compile(r"^branch(\d+)_seg(\d+)$")

# What the solver input calls a kind of condition, as the boundary conditions table calls it.
KIND_LABELS = {"FLOW": bcs.INFLOW, "RCR": bcs.RCR_KIND, "RESISTANCE": bcs.RESISTANCE_KIND,
               "PRESSURE": "Pressure"}

# A vessel's element values, as the solver input names them, with what to call them and their
# units: cgs, which the package writes them in.
VESSEL_VALUES = (("R_poiseuille", "R", "dyn·s/cm⁵"), ("C", "C", "cm⁵/dyn"), ("L", "L", "g/cm⁴"),
                 ("stenosis_coefficient", "stenosis", "dyn·s²/cm⁸"))
CONDITION_UNITS = {"R": "dyn·s/cm⁵", "Rp": "dyn·s/cm⁵", "Rd": "dyn·s/cm⁵", "C": "cm⁵/dyn",
                   "Pd": "dyn/cm²"}


@dataclass(frozen=True)
class Node:
    key: str
    kind: str
    """VESSEL, JUNCTION or BOUNDARY."""
    label: str
    """What it is called: a vessel's name, a junction's, or the face a condition is on."""
    short: str = ""
    """What a block drawn small has room for: its branch, and a letter for its segment if the
    branch has more than one -- `6` for `branch6_seg0` alone, `25b` for `branch25_seg1`. Letters
    rather than `25.1`, which reads as a number."""
    detail: str = ""
    """The kind of condition, for a condition."""
    vessel: str = ""
    """The vessel's name, for a vessel: what its results are filed under."""
    cap: str = ""
    """The face a condition is on, for a condition."""
    parameters: tuple = ()
    """(name, value, unit) for what the solver input gives the block."""


@dataclass(frozen=True)
class Network:
    nodes: dict
    """{key: Node}, conditions, then vessels, then junctions, each in the order the input has them."""
    edges: tuple
    """(from key, to key), the way the model was traced."""
    root: str | None
    """The source's condition, which the network is laid out from."""

    def children(self, key):
        return [target for source, target in self.edges if source == key]

    def parents(self, key):
        return [source for source, target in self.edges if target == key]

    def node_for_cap(self, cap):
        return next((node for node in self.nodes.values()
                     if node.kind == BOUNDARY and node.cap == cap), None)


def vessel_key(vessel_id) -> str:
    return f"{VESSEL}:{vessel_id}"


def junction_key(name) -> str:
    return f"{JUNCTION}:{name}"


def boundary_key(bc_name) -> str:
    return f"{BOUNDARY}:{bc_name}"


def network_from_config(config, inlet_name: str) -> Network:
    """The solver input's network.

    :param config: the solver input, as a dict or a path to it.
    :param inlet_name: the source's face, which the package's condition for it does not name --
      it is called `INFLOW` -- and which only the caller knows.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    nodes = {}
    for bc in config.get("boundary_conditions", []):
        bc_name, bc_type = bc["bc_name"], bc.get("bc_type", "")
        cap = cap_name(bc_name, bc_type, inlet_name)
        nodes[boundary_key(bc_name)] = Node(
            boundary_key(bc_name), BOUNDARY, cap, detail=KIND_LABELS.get(bc_type, bc_type), cap=cap,
            parameters=_condition_parameters(bc))
    segments = {}
    for vessel in config.get("vessels", []):
        match = _VESSEL_NAME.match(vessel["vessel_name"])
        if match:
            segments[match.group(1)] = segments.get(match.group(1), 0) + 1
    for vessel in config.get("vessels", []):
        values = vessel.get("zero_d_element_values", {})
        parameters = tuple((label, float(values[name]), unit) for name, label, unit in VESSEL_VALUES
                           if name in values)
        if "vessel_length" in vessel:
            parameters += (("length", float(vessel["vessel_length"]), "cm"),)
        key = vessel_key(vessel["vessel_id"])
        name = vessel["vessel_name"]
        nodes[key] = Node(key, VESSEL, name, short=_short_name(name, segments), vessel=name,
                          parameters=parameters)
    for junction in config.get("junctions", []):
        key = junction_key(junction["junction_name"])
        nodes[key] = Node(key, JUNCTION, junction["junction_name"])

    edges = []
    for vessel in config.get("vessels", []):
        key = vessel_key(vessel["vessel_id"])
        for end, bc_name in vessel.get("boundary_conditions", {}).items():
            condition = boundary_key(bc_name)
            if condition in nodes:
                edges.append((condition, key) if end == "inlet" else (key, condition))
    for junction in config.get("junctions", []):
        key = junction_key(junction["junction_name"])
        edges += [(vessel_key(i), key) for i in junction.get("inlet_vessels", [])
                  if vessel_key(i) in nodes]
        edges += [(key, vessel_key(o)) for o in junction.get("outlet_vessels", [])
                  if vessel_key(o) in nodes]

    root = boundary_key("INFLOW") if boundary_key("INFLOW") in nodes else None
    if root is None:
        targets = {target for _source, target in edges}
        root = next((key for key in nodes if key not in targets), None)
    return Network(nodes, tuple(edges), root)


def _short_name(name, segments) -> str:
    match = _VESSEL_NAME.match(name)
    if not match:
        return name
    branch, segment = match.group(1), int(match.group(2))
    if segments.get(branch, 1) == 1:
        return branch
    return f"{branch}{chr(ord('a') + segment)}" if segment < 26 else f"{branch}.{segment}"


def _condition_parameters(bc) -> tuple:
    """A condition's values as (name, value, unit), its waveform summed up rather than listed."""
    values = bc.get("bc_values", {})
    if "Q" in values and "t" in values:
        time, flow = np.asarray(values["t"], dtype=float), np.asarray(values["Q"], dtype=float)
        return (("points", len(time), ""), ("period", float(time[-1] - time[0]), "s"),
                ("mean Q", _cycle_mean(time, flow), "mL/s"))
    return tuple((name, float(value), CONDITION_UNITS.get(name, ""))
                 for name, value in values.items() if isinstance(value, (int, float)))


def layered_layout(network: Network, conditions_last: bool = True) -> dict:
    """{key: (layer, row)}: layers out from the source, rows so that no two leaves share one.

    A node's layer is how many connections it is from the source, so every edge inside the tree
    goes one layer on; each leaf has a row of its own, in the order the junctions list their
    outlets, and every other node sits midway between its first and last child, which is what
    keeps a branching readable without any edge crossing another. A part of the network the
    source does not reach, which a well-formed input has none of, is laid out below the rest from
    a root of its own rather than left off.

    With `conditions_last`, the conditions at the ends of the tree are moved to one last layer of
    their own, however far from the source each is: they are the blocks with long names, which
    scattered through the layers would widen every one of them, and in one column they read down
    the page as the conditions table does.
    """
    children = {key: [] for key in network.nodes}
    for source, target in network.edges:
        children[source].append(target)
    layer, row, parent = {}, {}, {}
    next_row = 0.0

    def lay_out(root):
        # Iteratively, depth first: a model with long branches cut into many segments is deep
        # enough for recursion to be a risk.
        nonlocal next_row
        stack = [(root, 0, False)]
        while stack:
            key, depth, finished = stack.pop()
            if finished:
                placed = [child for child in children[key] if parent.get(child) == key]
                if placed:
                    row[key] = (row[placed[0]] + row[placed[-1]]) / 2.0
                else:
                    row[key] = next_row
                    next_row += 1.0
                continue
            if key in layer:
                continue
            layer[key] = depth
            stack.append((key, depth, True))
            for child in reversed(children[key]):
                if child not in layer:
                    parent.setdefault(child, key)
                    stack.append((child, depth + 1, False))

    for root in ([network.root] if network.root else []) + list(network.nodes):
        if root not in layer:
            lay_out(root)
    if conditions_last:
        ends = [key for key, node in network.nodes.items()
                if node.kind == BOUNDARY and key != network.root and not children[key]]
        last = 1 + max((layer[key] for key in network.nodes if key not in ends), default=0)
        for key in ends:
            layer[key] = last
    return {key: (layer[key], row[key]) for key in network.nodes}


def mean_pressures(network: Network, vessel_results: dict, face_results=()) -> dict:
    """{key: cycle-averaged pressure in mmHg} for every node the results reach.

    A vessel's is the mean of its two ends, which is what the centerlines are coloured by at its
    middle; a condition's is its cap's; a junction's is the pressure at the end of the vessel
    going into it, which is the one pressure a junction has.
    """
    pressures = {}
    for key, node in network.nodes.items():
        if node.kind == VESSEL and node.vessel in vessel_results:
            result = vessel_results[node.vessel]
            pressures[key] = (_cycle_mean(result.time, result.pressure_in)
                              + _cycle_mean(result.time, result.pressure_out)) / 2.0 / bcs.MMHG
    for face in face_results:
        node = network.node_for_cap(face.name)
        if node is not None:
            pressures[node.key] = face.mean_pressure_mmhg
    for key, node in network.nodes.items():
        if node.kind == JUNCTION:
            inlet = next((network.nodes[source] for source in network.parents(key)), None)
            if inlet is not None and inlet.vessel in vessel_results:
                result = vessel_results[inlet.vessel]
                pressures[key] = _cycle_mean(result.time, result.pressure_out) / bcs.MMHG
    return pressures


def describe(network: Network, key: str, vessel_results=None, face_results=()) -> str:
    """What a block is and what it came out at, in a line or two, for the panel to show."""
    node = network.nodes[key]
    values = " · ".join(f"{name} {_number(value)}" + (f" {unit}" if unit else "")
                        for name, value, unit in node.parameters)
    if node.kind == BOUNDARY:
        text = f"{node.cap}: {node.detail}" + (f", {values}" if values else "")
        face = next((face for face in face_results if face.name == node.cap), None)
        if face is not None:
            low, high = face.pressure_range_mmhg
            text += (f"\nMean flow {face.mean_flow:.4g} mL/s · mean pressure "
                     f"{face.mean_pressure_mmhg:.4g} mmHg ({low:.4g} to {high:.4g})")
        return text
    if node.kind == JUNCTION:
        inlets, outlets = network.parents(key), network.children(key)
        text = (f"{node.label}: a junction, {len(inlets)} vessel in and {len(outlets)} out"
                if len(inlets) == 1 else
                f"{node.label}: a junction, {len(inlets)} vessels in and {len(outlets)} out")
        pressure = mean_pressures(network, vessel_results or {}).get(key)
        if pressure is not None:
            text += f"\nMean pressure {pressure:.4g} mmHg"
        return text
    text = f"{node.label}: {values}"
    result = (vessel_results or {}).get(node.vessel)
    if result is not None:
        flow = (_cycle_mean(result.time, result.flow_in) + _cycle_mean(result.time, result.flow_out)) / 2.0
        p_in = _cycle_mean(result.time, result.pressure_in) / bcs.MMHG
        p_out = _cycle_mean(result.time, result.pressure_out) / bcs.MMHG
        text += f"\nMean flow {flow:.4g} mL/s · mean pressure {p_in:.4g} → {p_out:.4g} mmHg"
    return text


def _number(value) -> str:
    return f"{value:d}" if isinstance(value, int) else f"{value:.4g}"
