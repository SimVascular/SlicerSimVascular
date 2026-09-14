"""Turning upstream labels into face names.

A clip point label is free text and a face name is a file name, so something has to reduce one
to the other. It sits in the package rather than in the panel so that the headless workflow
scripts get the same rule: a case packaged from a terminal and a case packaged from the panel
have to produce the same `mesh-surfaces/` file names, or results from the two cannot be compared.
"""

import pytest

from svmeshcomplete.face_table import (
    CAP_PREFIX,
    WALL_NAME,
    Face,
    FaceTable,
    cap_name,
    names_from_labels,
    sanitized,
)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("RSVC", "RSVC"),
        ("Outlet 1", "Outlet_1"),
        ("Outlet  1", "Outlet_1"),          # a run collapses to one separator, not two
        ("left / right", "left_right"),
        ("aorta (distal)", "aorta_distal"), # trailing junk does not leave a trailing underscore
        ("  Inlet  ", "Inlet"),
        ("lpa-a", "lpa-a"),                 # already safe, and left alone
        ("v1.2", "v1.2"),
        ("", ""),
        ("   ", ""),
        ("///", ""),                        # nothing safe in it at all
        (None, ""),
    ],
)
def test_a_label_is_reduced_to_what_a_file_name_may_contain(label, expected):
    assert sanitized(label) == expected


def test_a_cap_name_is_the_label_behind_the_prefix():
    assert cap_name("Outlet 1") == "cap_Outlet_1"
    assert cap_name("RSVC") == "cap_RSVC"


def test_a_label_that_says_nothing_yields_no_name():
    """Not `cap_` on its own, and not a placeholder either.

    A face with no usable label has to come out unnamed, so that it shows up as one of the faces
    still to name. A placeholder would look like a name and be exported as one.
    """
    assert cap_name("") == ""
    assert cap_name("!!!") == ""


def test_the_wall_is_named_without_a_label():
    """Upstream the wall is not a vessel end and carries no label, so its name is not derived."""
    derived = names_from_labels({2: "RSVC"}, wall_face_id=1)
    assert derived.names == {1: WALL_NAME, 2: "cap_RSVC"}
    assert derived.notes == ()


def test_names_are_derived_for_every_labelled_face():
    derived = names_from_labels({2: "Inlet", 3: "Outlet 1", 4: "Outlet 2"}, wall_face_id=1)
    assert derived.names == {
        1: "wall", 2: "cap_Inlet", 3: "cap_Outlet_1", 4: "cap_Outlet_2"
    }
    assert derived.notes == ()


def test_duplicate_labels_are_numbered_apart_rather_than_refused():
    """Two ends can carry the same label honestly, and an export must not be blocked on it.

    FaceTable requires unique names, and rightly: they become file names. But refusing here
    would stop an export on names nobody typed, with nothing to fix in this panel. Every member
    of the colliding group takes its face id, which is unique and says where it came from.
    """
    derived = names_from_labels({2: "Outlet 1", 3: "Outlet 1", 4: "IVC"}, wall_face_id=1)
    assert derived.names == {
        1: "wall", 2: "cap_Outlet_1_2", 3: "cap_Outlet_1_3", 4: "cap_IVC"
    }
    assert len(derived.notes) == 1
    assert "cap_Outlet_1_2" in derived.notes[0]
    assert "cap_Outlet_1_3" in derived.notes[0]

    # And the result is a table FaceTable will take, which is the point of resolving it here.
    FaceTable([Face(face_id, name) for face_id, name in derived.names.items()])


def test_labels_that_sanitize_together_collide_too():
    """The collision need not be in the labels: sanitizing can create one.

    `Outlet 1` and `Outlet@1` are different labels upstream and the same file name here, which
    is exactly the kind of thing that would otherwise surface as a failed export.
    """
    derived = names_from_labels({2: "Outlet 1", 3: "Outlet@1"}, wall_face_id=1)
    assert sorted(derived.names.values()) == ["cap_Outlet_1_2", "cap_Outlet_1_3", "wall"]
    assert derived.notes


def test_a_face_with_no_usable_label_is_reported_and_left_unnamed():
    derived = names_from_labels({2: "RSVC", 3: ""}, wall_face_id=1)
    assert derived.names == {1: "wall", 2: "cap_RSVC"}
    assert 3 not in derived.names
    assert len(derived.notes) == 1
    assert "3" in derived.notes[0]


def test_no_labels_at_all_derives_nothing():
    """Falsy, so a host can ask "is there anything upstream" without a second call."""
    derived = names_from_labels({})
    assert derived.names == {}
    assert not derived


def test_the_wall_may_be_the_only_thing_known():
    """A mesh whose wall id was recorded but whose caps were not still gets its wall named."""
    derived = names_from_labels({}, wall_face_id=1)
    assert derived.names == {1: "wall"}
    assert derived


def test_a_label_on_the_wall_face_id_does_not_beat_the_wall_name():
    """The wall is the wall. Upstream has no business naming it, and if it does it is ignored.

    Guards the case where a host passes the whole recorded map without taking the wall out of
    it: naming the wall `cap_something` would put it in `mesh-surfaces/` as a face flow crosses.
    """
    derived = names_from_labels({1: "Inlet", 2: "RSVC"}, wall_face_id=1)
    assert derived.names[1] == WALL_NAME
    assert not derived.names[1].startswith(CAP_PREFIX)
