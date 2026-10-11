import json

import pytest

from svromsetup import boundary_conditions as bcs
from svromsetup.boundary_conditions import RCR, BoundaryConditionError, Inflow, Resistance

CAPS = {2: "cap_inlet", 3: "cap_right", 4: "cap_left"}


# -- typed in the table ----------------------------------------------------------
@pytest.mark.parametrize("condition", [Inflow.steady(12.5), RCR(100.0, 1e-4, 1500.0, 0.0),
                                       Resistance(250.0)])
def test_what_the_table_shows_reads_back_as_the_same_condition(condition):
    text = bcs.to_text(condition)
    assert bcs.from_text(condition.kind, text) == condition


def test_an_rcr_distal_pressure_may_be_left_out_and_a_resistance_has_none():
    """A resistance is R alone, as svMultiPhysics' is: a second value is refused rather than
    read as a distal pressure the 3D solver would then drop."""
    assert bcs.from_text(bcs.RCR_KIND, "100, 1e-4 1500") == RCR(100.0, 1e-4, 1500.0, 0.0)
    assert bcs.from_text(bcs.RESISTANCE_KIND, "250") == Resistance(250.0)
    with pytest.raises(BoundaryConditionError, match="takes R; 2 value"):
        bcs.from_text(bcs.RESISTANCE_KIND, "250, 6666")


def test_a_waveform_survives_its_own_description_and_is_replaced_by_a_number():
    """The table hands back the cell's text whether it was edited or not, and a waveform's text
    is a description that cannot be parsed back into one. Left alone it has to stay the waveform;
    typed over with a number, it becomes a steady flow."""
    waveform = Inflow((0.0, 0.4, 0.8), (5.0, 20.0, 5.0))
    text = bcs.to_text(waveform)
    assert "waveform" in text
    assert bcs.from_text(bcs.INFLOW, text, previous=waveform) is waveform
    assert bcs.from_text(bcs.INFLOW, "7", previous=waveform, period=0.8) == Inflow.steady(7.0, 0.8)


def test_a_waveform_read_from_a_file_is_shown_as_that_file(tmp_path):
    """The cell says which file a cap's inflow is, and leaving it as it is keeps the waveform."""
    path = tmp_path / "cap_RSVC.flow"
    path.write_text("0.0 -5.0\n0.4 -15.0\n0.8 -5.0\n")
    waveform, turned = bcs.as_inflow(bcs.read_flow_file(path))
    assert turned
    assert bcs.to_text(waveform) == str(path.resolve()), "turned round, and still that file's"
    assert bcs.from_text(bcs.INFLOW, str(path.resolve()), previous=waveform) is waveform
    # The file is where it came from, not what it is: the same numbers are the same inflow.
    assert waveform == Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0))
    # A steady flow read from a file is still that file's, rather than a number nobody typed.
    (tmp_path / "steady.flow").write_text("0.0 3.0\n0.8 3.0\n")
    assert bcs.to_text(bcs.read_flow_file(tmp_path / "steady.flow")).endswith("steady.flow")
    # And the file is kept with the scene, so it is shown again after reopening it.
    assert bcs.from_json(bcs.to_json({3: waveform}))[3].source == str(path.resolve())


@pytest.mark.parametrize("kind, text", [(bcs.RCR_KIND, "100, 200"), (bcs.RESISTANCE_KIND, "1 2 3"),
                                        (bcs.INFLOW, "fast"), (bcs.RCR_KIND, "1, nan, 3")])
def test_values_that_are_not_a_condition_say_what_was_expected(kind, text):
    with pytest.raises(BoundaryConditionError, match="takes|numbers|finite"):
        bcs.from_text(kind, text)


def test_an_empty_cell_is_no_condition_rather_than_an_error():
    assert bcs.from_text(bcs.RCR_KIND, "  ") is None


def test_a_waveform_has_to_be_one_cycle_from_zero():
    with pytest.raises(BoundaryConditionError, match="two points"):
        Inflow((0.0,), (1.0,))
    with pytest.raises(BoundaryConditionError, match="start at time 0"):
        Inflow((0.1, 1.0), (1.0, 1.0))
    with pytest.raises(BoundaryConditionError, match="increase"):
        Inflow((0.0, 0.5, 0.5), (1.0, 1.0, 1.0))


def test_the_mean_of_a_waveform_is_over_time_not_over_points():
    # Most of the points crowd the first tenth of the cycle; a mean over points would read 9.
    inflow = Inflow((0.0, 0.025, 0.05, 0.075, 0.1, 1.0), (10.0, 10.0, 10.0, 10.0, 10.0, 0.0))
    assert inflow.mean == pytest.approx(0.1 * 10 + 0.9 * 5.0)


# -- kept on the mesh ----------------------------------------------------------------
def test_the_conditions_survive_being_saved_and_a_bad_entry_is_dropped_not_raised():
    conditions = {2: Inflow((0.0, 0.5, 1.0), (1.0, 3.0, 1.0)), 3: RCR(1.0, 2.0, 3.0, 4.0),
                  4: Resistance(5.0)}
    text = bcs.to_json(conditions)
    assert bcs.from_json(text) == conditions
    assert bcs.to_json(conditions) == text, "saving twice untouched should be the same bytes"

    stored = json.loads(text)
    stored["3"] = {"type": "RCR", "Rp": "not a number"}
    stored["9"] = {"type": "Something else"}
    assert bcs.from_json(json.dumps(stored)) == {2: conditions[2], 4: conditions[4]}
    assert bcs.from_json("not json") == {}


# -- whether they close the model ---------------------------------------------------------
def test_a_complete_set_has_no_problems():
    conditions = {2: Inflow.steady(10.0), 3: Resistance(100.0), 4: Inflow.steady(2.0)}
    assert bcs.problems(conditions, CAPS, 2) == []


def test_every_problem_is_named():
    assert "cap_left" in " ".join(bcs.problems({2: Inflow.steady(1.0), 3: Resistance(1.0)}, CAPS, 2))
    assert "has to be an inflow" in " ".join(
        bcs.problems({2: Resistance(1.0), 3: Resistance(1.0), 4: Resistance(1.0)}, CAPS, 2))
    assert "nothing sets the pressure" in " ".join(
        bcs.problems({2: Inflow.steady(1.0), 3: Inflow.steady(1.0), 4: Inflow.steady(1.0)}, CAPS, 2))
    assert "source" in " ".join(bcs.problems({}, CAPS, None))
    periods = bcs.problems({2: Inflow((0.0, 0.4, 0.8), (1.0, 2.0, 1.0)),
                            3: Inflow((0.0, 0.5, 1.0), (1.0, 2.0, 1.0)), 4: Resistance(1.0)}, CAPS, 2)
    assert "different periods" in " ".join(periods)


def test_a_steady_inflow_takes_the_period_of_the_waveforms_beside_it():
    waveform = Inflow((0.0, 0.4, 0.8), (1.0, 2.0, 1.0))
    period = bcs.cycle_period([Inflow.steady(3.0), waveform, Resistance(1.0)])
    assert period == 0.8
    assert bcs.with_period(Inflow.steady(3.0), period) == Inflow.steady(3.0, 0.8)
    assert bcs.with_period(waveform, 2.0) is waveform
    assert bcs.cycle_period([Inflow.steady(3.0)]) == 1.0


# -- SimVascular's files -------------------------------------------------------------------
def test_rcrt_and_resistance_files_round_trip(tmp_path):
    rcrs = {"cap_right": RCR(100.0, 1e-4, 1500.0, 10.0), "cap_left": RCR(200.0, 2e-4, 2500.0, 0.0)}
    assert bcs.read_rcrt(bcs.write_rcrt(tmp_path / "rcrt.dat", rcrs)) == rcrs
    resistances = {"cap_right": Resistance(100.0), "cap_left": Resistance(200.0)}
    assert bcs.read_resistance(bcs.write_resistance(tmp_path / "resistance.dat", resistances)) \
        == resistances


def test_a_simvascular_resistance_with_a_distal_pressure_is_refused_by_name(tmp_path):
    path = tmp_path / "resistance.dat"
    path.write_text("cap_lpa 1200 0\ncap_rpa 900\n")
    assert bcs.read_resistance(path) == {"cap_lpa": Resistance(1200.0), "cap_rpa": Resistance(900.0)}
    path.write_text("cap_lpa 1200 6666\n")
    with pytest.raises(BoundaryConditionError, match="cap_lpa.*RCR"):
        bcs.read_resistance(path)


def test_an_rcrt_written_by_simvascular_is_read(tmp_path):
    path = tmp_path / "rcrt.dat"
    path.write_text("2\n2\ncap_aorta\n121\n0.000155\n1212\n0.0 0\n1.0 0\n"
                    "2\ncap_carotid\n500\n0.0001\n9000\n0.0 0\n1.0 0\n")
    assert bcs.read_rcrt(path) == {"cap_aorta": RCR(121.0, 0.000155, 1212.0, 0.0),
                                   "cap_carotid": RCR(500.0, 0.0001, 9000.0, 0.0)}


def test_an_rcrt_with_a_varying_distal_pressure_is_refused_by_name(tmp_path):
    path = tmp_path / "rcrt.dat"
    path.write_text("2\n2\ncap_aorta\n121\n0.000155\n1212\n0.0 0\n1.0 10\n")
    with pytest.raises(BoundaryConditionError, match="cap_aorta"):
        bcs.read_rcrt(path)


def test_a_flow_file_round_trips_and_a_negative_one_is_turned_round_and_said_to_be(tmp_path):
    inflow = Inflow((0.0, 0.5, 1.0), (4.0, 9.0, 4.0))
    assert bcs.read_flow_file(bcs.write_flow_file(tmp_path / "inflow.flow", inflow)) == inflow

    # svMultiPhysics' convention: inflow is negative, along the outward normal.
    (tmp_path / "3d.flow").write_text("0.0 -4.0\n0.5 -9.0\n1.0 -4.0\n")
    turned, was_turned = bcs.as_inflow(bcs.read_flow_file(tmp_path / "3d.flow"))
    assert was_turned and turned == inflow
    assert bcs.as_inflow(inflow) == (inflow, False)

    (tmp_path / "bad.flow").write_text("0.0 1.0\n0.5\n")
    with pytest.raises(BoundaryConditionError, match="line 2"):
        bcs.read_flow_file(tmp_path / "bad.flow")


def test_the_package_reads_the_files_as_they_are_written(tmp_path):
    """The files are written for `svromutils` to read, so its own reader is the judge."""
    pytest.importorskip("vmtk")
    io_1d = pytest.importorskip("svromutils.io_1d")
    from svromutils.parameters import Parameters

    bcs.write_rcrt(tmp_path / "rcrt.dat", {"cap_right": RCR(100.0, 1e-4, 1500.0, 10.0)})
    bcs.write_resistance(tmp_path / "resistance.dat", {"cap_left": Resistance(200.0)})
    parameters = Parameters()
    parameters.outflow_bc_type = ["rcrt.dat", "resistance.dat"]
    parameters.outflow_bc_file = str(tmp_path)
    values, kinds = io_1d.read_variable_outflow_bcs(parameters)
    assert values == {"cap_right": [100.0, 1e-4, 1500.0, 10.0], "cap_left": ["200", 0.0]}
    assert kinds == {"cap_right": "rcr", "cap_left": "resistance"}
