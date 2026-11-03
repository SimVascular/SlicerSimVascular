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
    assert bcs.to_text(waveform) == "cap_RSVC.flow", "turned round, and still that file's"
    assert bcs.from_text(bcs.INFLOW, "cap_RSVC.flow", previous=waveform) is waveform
    # The file is where it came from, not what it is: the same numbers are the same inflow.
    assert waveform == Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0))
    # A steady flow read from a file is still that file's, rather than a number nobody typed.
    (tmp_path / "steady.flow").write_text("0.0 3.0\n0.8 3.0\n")
    assert bcs.to_text(bcs.read_flow_file(tmp_path / "steady.flow")).endswith("steady.flow")
    # And the file is kept with the scene, so it is shown again after reopening it -- by its name,
    # and not its folder, which can carry a patient's name or ID.
    assert bcs.from_json(bcs.to_json({3: waveform}))[3].source == "cap_RSVC.flow"
    assert str(tmp_path) not in bcs.to_json({3: waveform})


def test_a_scene_saved_with_whole_paths_keeps_only_the_files_names():
    """Scenes saved before the folder was dropped kept the whole path: read, and saved again, they
    keep the file's name alone, a Windows path's as much as any other's."""
    for path in ("/Users/someone/MRN123/cap_RSVC.flow", "C:\\Patients\\MRN123\\cap_RSVC.flow"):
        saved = json.dumps({"3": {"type": bcs.INFLOW, "t": [0.0, 1.0], "Q": [2.0, 2.0], "file": path}})
        read = bcs.from_json(saved)[3]
        assert read.source == "cap_RSVC.flow"
        assert "MRN123" not in bcs.to_json({3: read})


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


def test_a_flow_file_written_for_svmultiphysics_is_read_without_its_header(tmp_path):
    """Its `points modes` line skipped, and nothing else: rows whose times are whole numbers,
    written without a decimal point, are rows, the first and last among them."""
    (tmp_path / "3d.flow").write_text("3 16\n0 -4\n0.5 -9\n1 -4\n")
    assert bcs.read_flow_file(tmp_path / "3d.flow") == Inflow((0.0, 0.5, 1.0), (-4.0, -9.0, -4.0))
    (tmp_path / "plain.flow").write_text("0 4\n0.5 9\n1 4\n")
    assert bcs.read_flow_file(tmp_path / "plain.flow") == Inflow((0.0, 0.5, 1.0), (4.0, 9.0, 4.0))


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


# -- a .flow as a table, and an inflow inverted ---------------------------------------------
def test_a_flow_file_is_read_as_its_two_columns_whatever_they_hold(tmp_path):
    path = tmp_path / "cap.flow"
    path.write_text("# a comment\n0.0\t-5.0\r\n0.4\t-15.0\r\n0.8\t-5.0\r\n")
    assert bcs.read_flow_columns(path) == ([0.0, 0.4, 0.8], [-5.0, -15.0, -5.0])
    # svMultiPhysics' header is not a row: 3 points, 16 Fourier coefficients.
    path.write_text("3 16\n0.0 -5.0\n0.4 -15.0\n0.8 -5.0\n")
    assert bcs.read_flow_columns(path) == ([0.0, 0.4, 0.8], [-5.0, -15.0, -5.0])
    # Nothing asked of the numbers, as a table shows them: not from 0, not increasing.
    path.write_text("0.5 1\n0.2 2\n")
    assert bcs.read_flow_columns(path) == ([0.5, 0.2], [1.0, 2.0])
    path.write_text("0.0 1\nfoo bar\n")
    with pytest.raises(BoundaryConditionError, match="line 2: expected a time and a flow"):
        bcs.read_flow_columns(path)


def test_an_inverted_inflow_says_so_and_keeps_saying_so(tmp_path):
    path = tmp_path / "inflow.flow"
    path.write_text("0.0 -5.0\n0.4 -15.0\n0.8 -5.0\n")
    inverted = bcs.read_flow_file(path).flipped()
    assert inverted.negated and inverted.flow == (5.0, 15.0, 5.0)
    assert not inverted.flipped().negated, "inverted twice is the file as it is"
    back = bcs.from_json(bcs.to_json({2: inverted}))[2]
    assert back.negated and back == inverted and back.source == inverted.source


def test_a_condition_is_typed_one_value_to_a_box():
    """By name, in the order the boxes are labelled; the missing named, the unreadable named."""
    assert bcs.from_fields(bcs.RCR_KIND, ["100", "1e-4", "1500"]) == RCR(100.0, 1e-4, 1500.0, 0.0)
    assert bcs.from_fields(bcs.RCR_KIND, ["100", "1e-4", "1500", "1333.22"]) == RCR(100.0, 1e-4, 1500.0, 1333.22)
    assert bcs.from_fields(bcs.RESISTANCE_KIND, [" 250 "]) == Resistance(250.0)
    assert bcs.from_fields(bcs.INFLOW, ["-7"], period=0.8) == Inflow.steady(-7.0, 0.8)

    assert bcs.from_fields(bcs.RCR_KIND, ["100", "", "1500"]) is None, "incomplete is not a condition"
    assert bcs.missing_fields(bcs.RCR_KIND, ["100", "", "1500"]) == ["C"]
    assert bcs.missing_fields(bcs.RCR_KIND, ["1", "2", "3"]) == [], "Pd may be left out"
    assert bcs.from_fields(bcs.INFLOW, []) is None
    for fields, name in ((["100", "1,5e-4", "1500"], "C"), (["inf"], "R")):
        kind = bcs.RCR_KIND if len(fields) == 3 else bcs.RESISTANCE_KIND
        with pytest.raises(bcs.FieldError) as raised:
            bcs.from_fields(kind, fields)
        assert raised.value.name == name


def test_a_condition_is_shown_in_its_boxes_as_it_reads_back():
    """Every value as the number it is, not six figures of it: a box is read back whenever another
    of the same condition's is edited."""
    rcr = RCR(121.0, 1.234567891e-4, 1212.0, 0.0)
    assert bcs.fields_of(rcr) == ["121", "0.0001234567891", "1212", "0"]
    assert bcs.from_fields(bcs.RCR_KIND, bcs.fields_of(rcr)) == rcr
    assert bcs.fields_of(Inflow.steady(-12.0)) == ["-12"]
    assert bcs.fields_of(Inflow((0.0, 0.5, 1.0), (1.0, 2.0, 1.0))) == [""], "a waveform has no box"
    assert bcs.fields_of(None) == []
