import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from svromsetup import network, results
from svromsetup.boundary_conditions import MMHG
from svromsetup.solver import VesselResult

# A source feeding two branches, one of them cut into two segments, and an inflow prescribed at
# that branch's far end: everything a network from the package has, in the smallest one.
CONFIG = {
    "boundary_conditions": [
        {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"t": [0.0, 0.5, 1.0], "Q": [5.0, 15.0, 5.0]}},
        {"bc_name": "RCR_cap_right", "bc_type": "RCR",
         "bc_values": {"Rp": 100.0, "C": 1e-4, "Rd": 1500.0, "Pd": 0.0}},
        {"bc_name": "FLOW_cap_left", "bc_type": "FLOW", "bc_values": {"t": [0.0, 1.0], "Q": [-2.0, -2.0]}},
    ],
    "vessels": [
        {"vessel_id": 0, "vessel_name": "branch0_seg0", "vessel_length": 2.0,
         "boundary_conditions": {"inlet": "INFLOW"},
         "zero_d_element_values": {"R_poiseuille": 1.5, "C": 2e-5, "L": 0.3, "stenosis_coefficient": 0.0}},
        {"vessel_id": 1, "vessel_name": "branch1_seg0", "vessel_length": 3.0,
         "boundary_conditions": {"outlet": "RCR_cap_right"},
         "zero_d_element_values": {"R_poiseuille": 9.0, "C": 1e-5, "L": 1.1}},
        {"vessel_id": 2, "vessel_name": "branch2_seg0", "vessel_length": 1.0,
         "zero_d_element_values": {"R_poiseuille": 4.0, "C": 1e-6, "L": 0.9}},
        {"vessel_id": 3, "vessel_name": "branch2_seg1", "vessel_length": 3.0,
         "boundary_conditions": {"outlet": "FLOW_cap_left"},
         "zero_d_element_values": {"R_poiseuille": 12.0, "C": 3e-6, "L": 2.7}},
    ],
    "junctions": [
        {"junction_name": "J0", "junction_type": "NORMAL_JUNCTION", "inlet_vessels": [0], "outlet_vessels": [1, 2]},
        {"junction_name": "J1", "junction_type": "NORMAL_JUNCTION", "inlet_vessels": [2], "outlet_vessels": [3]},
    ],
}


@pytest.fixture
def net():
    return network.network_from_config(CONFIG, "cap_inlet")


def test_every_block_is_a_node_and_every_connection_an_edge_traced_from_the_source(net):
    kinds = [node.kind for node in net.nodes.values()]
    assert kinds.count(network.BOUNDARY) == 3 and kinds.count(network.VESSEL) == 4
    assert kinds.count(network.JUNCTION) == 2
    assert net.root == "boundary:INFLOW"
    assert set(net.edges) == {
        ("boundary:INFLOW", "vessel:0"), ("vessel:0", "junction:J0"),
        ("junction:J0", "vessel:1"), ("junction:J0", "vessel:2"), ("vessel:1", "boundary:RCR_cap_right"),
        ("vessel:2", "junction:J1"), ("junction:J1", "vessel:3"), ("vessel:3", "boundary:FLOW_cap_left")}


def test_conditions_are_named_after_their_faces_and_kinds_as_the_table_names_them(net):
    assert net.nodes["boundary:INFLOW"].label == "cap_inlet"
    assert net.node_for_cap("cap_left").key == "boundary:FLOW_cap_left"
    assert [net.nodes[k].detail for k in ("boundary:INFLOW", "boundary:RCR_cap_right")] == ["Inflow", "RCR"]
    # An inflow prescribed at a far end is an edge out of the model, the way it was traced.
    assert ("vessel:3", "boundary:FLOW_cap_left") in net.edges


def test_the_layout_goes_one_layer_per_edge_and_centres_each_node_on_its_children(net):
    positions = network.layered_layout(net, conditions_last=False)
    assert positions[net.root][0] == 0
    for source, target in net.edges:
        assert positions[target][0] == positions[source][0] + 1
    leaves = [key for key in net.nodes if not net.children(key)]
    rows = [positions[key][1] for key in leaves]
    assert len(set(rows)) == len(rows), "every leaf has a row of its own"
    for key in net.nodes:
        kids = net.children(key)
        if kids:
            assert positions[key][1] == pytest.approx((positions[kids[0]][1] + positions[kids[-1]][1]) / 2)


def test_the_conditions_at_the_ends_share_the_last_layer_and_keep_their_rows(net):
    """However far from the source each cap is: cap_right is two connections nearer than
    cap_left, and both are drawn in the last column, in the order the tree reaches them."""
    loose, lined_up = network.layered_layout(net, conditions_last=False), network.layered_layout(net)
    ends = ("boundary:RCR_cap_right", "boundary:FLOW_cap_left")
    assert loose[ends[0]][0] < loose[ends[1]][0]
    assert lined_up[ends[0]][0] == lined_up[ends[1]][0] == 1 + max(
        lined_up[key][0] for key in net.nodes if key not in ends)
    assert all(lined_up[key][1] == loose[key][1] for key in net.nodes)
    assert lined_up[net.root][0] == 0, "the source's condition stays first"
    for source, target in net.edges:
        assert lined_up[target][0] > lined_up[source][0]


def test_a_vessel_drawn_small_is_its_branch_and_a_letter_for_its_segment_if_it_has_several(net):
    assert [net.nodes[f"vessel:{i}"].short for i in range(4)] == ["0", "1", "2a", "2b"]


def test_a_part_the_source_does_not_reach_is_laid_out_rather_than_dropped():
    config = dict(CONFIG, vessels=CONFIG["vessels"] + [
        {"vessel_id": 9, "vessel_name": "branch9_seg0", "vessel_length": 1.0}])
    positions = network.layered_layout(network.network_from_config(config, "cap_inlet"))
    assert "vessel:9" in positions


def results_for(config):
    time = np.linspace(0.0, 1.0, 11)
    solved = {}
    for index, vessel in enumerate(config["vessels"]):
        p_in, p_out = (100.0 - 2 * index) * MMHG, (99.0 - 2 * index) * MMHG
        solved[vessel["vessel_name"]] = VesselResult(time, np.full(11, 3.0), np.full(11, 3.0),
                                                     np.full(11, p_in), np.full(11, p_out))
    return solved


def test_pressures_and_descriptions_say_what_the_block_holds_and_came_out_at(net):
    solved = results_for(CONFIG)
    faces = results.face_results(CONFIG, solved, "cap_inlet")
    pressures = network.mean_pressures(net, solved, faces)
    assert pressures["vessel:0"] == pytest.approx(99.5)
    assert pressures["junction:J0"] == pytest.approx(99.0), "the end of the vessel going in"
    assert pressures["boundary:RCR_cap_right"] == pytest.approx(97.0)

    vessel = network.describe(net, "vessel:1", solved, faces)
    assert vessel.startswith("branch1_seg0: R 9 dyn·s/cm⁵")
    assert "98 → 97 mmHg" in vessel and "length 3 cm" in vessel
    condition = network.describe(net, "boundary:RCR_cap_right", solved, faces)
    assert condition.startswith("cap_right: RCR, Rp 100") and "mean pressure 97 mmHg" in condition
    assert "mean Q 10 mL/s" in network.describe(net, "boundary:INFLOW")
    assert "1 vessel in and 2 out" in network.describe(net, "junction:J0", solved)


def centerline_along_x(branch_id, count=11):
    """One straight line of points x = 0, 1, ..., every one in the branch, its Path its x."""
    points = vtk.vtkPoints()
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(count)
    for index in range(count):
        points.InsertNextPoint(float(index), 0.0, 0.0)
        line.GetPointIds().SetId(index, index)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    polydata.SetLines(cells)
    for name, values in (("BranchId", np.full(count, branch_id)), ("Path", np.arange(count, dtype=float)),
                         ("MaximumInscribedSphereRadius", np.full(count, 0.5))):
        array = numpy_to_vtk(values.astype(float), deep=True)
        array.SetName(name)
        polydata.GetPointData().AddArray(array)
    return polydata


def test_a_vessel_is_picked_out_on_the_stretch_of_centerline_its_results_are_drawn_on():
    """Branch 2's two segments, one long and three long, share its centerline a quarter to three
    quarters -- the same split `centerline_results` colours by."""
    centerline = centerline_along_x(2)
    first = results.vessel_lines(centerline, CONFIG, "branch2_seg0")
    second = results.vessel_lines(centerline, CONFIG, "branch2_seg1")
    first_x = vtk_to_numpy(first.GetPoints().GetData())[:, 0]
    second_x = vtk_to_numpy(second.GetPoints().GetData())[:, 0]
    assert sorted(first_x) == [0.0, 1.0, 2.0] and first.GetNumberOfLines() == 1
    assert sorted(second_x) == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert first.GetPointData().GetArray("MaximumInscribedSphereRadius").GetValue(0) == 0.5

    coloured = results.centerline_results(centerline, CONFIG, results_for(CONFIG))
    pressure = vtk_to_numpy(coloured.GetPointData().GetArray(results.PRESSURE_ARRAY_NAME))
    assert pressure[1] > 95.0 > pressure[5], "branch2_seg0 and branch2_seg1 where vessel_lines says"
