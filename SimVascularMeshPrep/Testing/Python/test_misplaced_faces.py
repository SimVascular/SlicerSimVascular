"""Checking an inherited name against where the face actually is.

The one check on a record that is worth anything. Everything else about it can be verified
without leaving the numbers - the ids are on the mesh, each names a real control point, the names
are unique - and all of that stays true when the ids have been permuted, which has happened
upstream: a boundary layer rotated them and every id-based check passed throughout. What a
permutation cannot survive is being asked where the cap is.
"""

import pytest

from svmeshcomplete.faces import misplaced_faces


def test_faces_where_the_record_says_they_are_are_not_reported():
    centroids = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 4: (0.0, 10.0, 0.0)}
    assert misplaced_faces(centroids, centroids) == {}


def test_a_small_offset_is_not_a_misplacement():
    """A cap is remade on the inner surface of a boundary layer, so it never sits exactly on its
    clip point. What matters is which end it is nearest, not how near."""
    centroids = {2: (0.4, 0.0, 0.0), 3: (10.3, 0.0, 0.0)}
    expected = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0)}
    assert misplaced_faces(centroids, expected) == {}


def test_a_swap_is_reported_both_ways():
    """The fault that reads as correct everywhere except in space."""
    centroids = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0)}
    expected = {2: (10.0, 0.0, 0.0), 3: (0.0, 0.0, 0.0)}
    assert misplaced_faces(centroids, expected) == {2: 3, 3: 2}


def test_a_rotation_is_reported():
    """Three faces rotated, which is what the boundary layer actually did.

    Read the result as "face 2 sits where face 4 was supposed to be": the record here puts face
    2's end at face 3's position, so the face carrying id 2 is found where face 4's end is. That
    is the inverse of the rotation the record applied, which is what a reader of the panel needs -
    they are looking at face 2 and asking what it really is.
    """
    positions = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 4: (0.0, 10.0, 0.0)}
    expected = {2: positions[3], 3: positions[4], 4: positions[2]}
    assert misplaced_faces(positions, expected) == {2: 4, 3: 2, 4: 3}


def test_one_face_out_of_several_is_reported_alone():
    positions = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 4: (0.0, 10.0, 0.0)}
    expected = dict(positions)
    expected[3], expected[4] = expected[4], expected[3]
    assert misplaced_faces(positions, expected) == {3: 4, 4: 3}


def test_matching_is_one_to_one():
    """Two faces cannot both be attributed to the same end.

    Taking each face's nearest end independently, both of these would come out as end 2 and the
    swap between them would go unreported - one of them would look right.
    """
    centroids = {2: (0.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}
    expected = {2: (1.0, 0.0, 0.0), 3: (0.0, 0.0, 0.0)}
    assert misplaced_faces(centroids, expected) == {2: 3, 3: 2}


def test_a_single_face_is_never_misplaced():
    """With one end there is no other end it could have been confused with."""
    assert misplaced_faces({2: (0.0, 0.0, 0.0)}, {2: (99.0, 0.0, 0.0)}) == {}


def test_faces_with_nothing_to_compare_against_are_left_out():
    """A face the record says nothing about, and a record entry with no face on the mesh.

    Both are reported elsewhere - one as still to name, the other as a face the mesh has not got -
    and neither is evidence about the faces that can be checked.
    """
    centroids = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 9: (50.0, 0.0, 0.0)}
    expected = {2: (0.0, 0.0, 0.0), 3: (10.0, 0.0, 0.0), 7: (99.0, 0.0, 0.0)}
    assert misplaced_faces(centroids, expected) == {}


def test_nothing_to_check_is_not_an_error():
    assert misplaced_faces({}, {}) == {}
    assert misplaced_faces({2: (0.0, 0.0, 0.0)}, {}) == {}
