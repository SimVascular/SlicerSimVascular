"""Which face id is which vessel, and what the solver calls it.

A mesher's face ids are numbers, and the boundary conditions differ per vessel -- one inflow
waveform per systemic vein, a resistance per pulmonary branch -- so somebody has to say which
cap is the azygous vein. This is that mapping, validated.

The names become the `mesh-surfaces/` file names and through them the solver's `Add_face`
and `Add_BC` names, so they are what results come back labelled with.

Where the names come *from* need not be a person typing. Clip Vessel labels each clip point,
and records on its output which control point named each face; a host that can follow that
record arrives here with a label per face already. `names_from_labels` is the step between the
two: a control point label is free text -- spaces, punctuation, whatever was typed -- and a
face name is a file name, so it has to be reduced to one, and duplicate labels upstream have
to come out as distinct names rather than as a validation failure at export time.

Convention `mesh_complete` relies on: `cap_*` is a face flow crosses, one boundary
condition each; `wall` or `wall_*` is vessel wall, merged into `walls_combined.vtp`.

Where the names are *kept* is the host's business, not this module's. The Mesh Prep panel
keeps the ones that were typed, and works the rest out from Clip Vessel's record on every
load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

CAP_PREFIX = "cap_"
WALL_PREFIX = "wall_"

# The name a single-face wall carries, which is also what a wall face id is named when it comes
# from upstream rather than from a person.
WALL_NAME = "wall"

# What a face name may contain. The names become file names under `mesh-surfaces/` and `Add_BC`
# names in `solver.xml`, so they are held to what is safe in both on every platform: no spaces,
# no separators, nothing a shell or an XML attribute would need quoting for. Runs of anything
# else collapse to one underscore.
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class FaceTableError(ValueError):
    """Raised when a face table is missing, malformed or inconsistent."""


def sanitized(label: str) -> str:
    """A label reduced to what a face name may contain, or `""` if nothing is left of it.

    Runs of unsafe characters collapse to a single underscore, so `Outlet 1` comes out
    `Outlet_1` rather than `Outlet__1`, and leading and trailing ones are dropped so that the
    result does not meet its prefix with a doubled separator. An empty result is returned as
    such rather than as a placeholder: a face whose label said nothing is better shown as
    unnamed, where somebody will name it, than given a name that only looks like one.
    """
    return _UNSAFE.sub("_", str(label or "")).strip("_")


def cap_name(label: str) -> str:
    """`cap_` + a label made safe, or `""` where the label says nothing."""
    stem = sanitized(label)
    return f"{CAP_PREFIX}{stem}" if stem else ""


@dataclass(frozen=True)
class DerivedNames:
    """Face names worked out from upstream labels, and what had to be done to get them.

    `notes` is for the host to show. Nothing in it is an error -- the names are usable -- but
    each line is something the operator would want to know before trusting a name they did not
    type, which is the whole risk of inheriting names: `cap_Outlet_1` read off a positional
    default looks exactly like a name somebody chose.
    """

    names: Mapping[int, str]
    notes: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.names)


def names_from_labels(labels_by_face_id, wall_face_id: int | None = None) -> DerivedNames:
    """Face names from upstream labels: `{face_id: label}` in, `{face_id: name}` out.

    The wall is named `wall` and is not expected in the labels -- upstream it is not a vessel
    end and has no label of its own.

    Duplicate names are resolved here rather than raised. Two vessel ends can carry the same
    label honestly: Clip Vessel's auto-detected defaults repeat once an operator has added
    points, and sanitizing brings `Outlet 1` and `Outlet-1` together in any case. FaceTable
    would refuse the pair, and refusing is the wrong answer this early -- it would block an
    export on names nobody typed, with nothing to fix but the labels in another module. Every
    member of a colliding group gets its face id appended instead, which is ugly and unique and
    says where it came from, and a note is returned saying so.

    :param labels_by_face_id: the label for each face id; a face with no entry, or an entry that
      sanitizes to nothing, gets no name at all rather than a made-up one.
    """
    labels = {int(face_id): label for face_id, label in dict(labels_by_face_id).items()}
    names: dict[int, str] = {}
    if wall_face_id is not None:
        names[int(wall_face_id)] = WALL_NAME
        labels.pop(int(wall_face_id), None)

    proposed = {face_id: cap_name(label) for face_id, label in sorted(labels.items())}
    proposed = {face_id: name for face_id, name in proposed.items() if name}

    collisions: dict[str, list[int]] = {}
    for face_id, name in proposed.items():
        collisions.setdefault(name, []).append(face_id)

    notes = []
    for name, faceIds in sorted(collisions.items()):
        if len(faceIds) == 1:
            names[faceIds[0]] = name
            continue
        for face_id in faceIds:
            names[face_id] = f"{name}_{face_id}"
        notes.append(
            f"{len(faceIds)} faces upstream are named {name!r}; they are numbered apart as "
            + ", ".join(names[face_id] for face_id in faceIds)
            + "."
        )

    unnamed = sorted(set(labels) - set(names))
    if unnamed:
        notes.append(
            "No usable name upstream for face(s) "
            + ", ".join(str(face_id) for face_id in unnamed)
            + "."
        )
    return DerivedNames(names=names, notes=tuple(notes))


@dataclass(frozen=True)
class Face:
    face_id: int
    name: str

    @property
    def is_cap(self) -> bool:
        return self.name.startswith(CAP_PREFIX)

    @property
    def is_wall(self) -> bool:
        # A one-face wall is usually just `wall`; a split one is `wall_LPA`.
        return self.name == WALL_NAME or self.name.startswith(WALL_PREFIX)


class FaceTable:
    """The faces of one clipped surface, by id."""

    def __init__(self, faces) -> None:
        self.faces = tuple(sorted(faces, key=lambda face: face.face_id))
        ids = [face.face_id for face in self.faces]
        if len(set(ids)) != len(ids):
            raise FaceTableError(f"Face ids are not unique: {sorted(ids)}")
        names = [face.name for face in self.faces]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise FaceTableError(
                "Face names become file names and boundary condition names, so they have "
                f"to be unique: {duplicates}"
            )
        unclassified = [f.name for f in self.faces if not (f.is_cap or f.is_wall)]
        if unclassified:
            raise FaceTableError(
                f"Every name must be 'wall' or start with '{CAP_PREFIX}' or "
                f"'{WALL_PREFIX}', so the wall can be told from the caps: {unclassified}"
            )
        if not self.caps:
            raise FaceTableError("No cap: flow has nowhere to enter or leave.")
        if not self.walls:
            raise FaceTableError("No wall face: nothing to apply no-slip to.")

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self):
        return iter(self.faces)

    def __repr__(self) -> str:
        return f"FaceTable({len(self.caps)} caps, {len(self.walls)} wall faces)"

    @property
    def caps(self) -> tuple[Face, ...]:
        return tuple(face for face in self.faces if face.is_cap)

    @property
    def walls(self) -> tuple[Face, ...]:
        return tuple(face for face in self.faces if face.is_wall)

    def names_by_id(self) -> dict[int, str]:
        return {face.face_id: face.name for face in self.faces}

    def name_of(self, face_id: int) -> str:
        try:
            return self.names_by_id()[face_id]
        except KeyError:
            raise FaceTableError(
                f"Face id {face_id} is not in the table, which has "
                f"{sorted(self.names_by_id())}."
            ) from None
