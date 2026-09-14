"""Which face id is which vessel, and what the solver calls it.

A mesher's face ids are numbers, and Clip Vessel's labels for them are positional --
`Outlet 14` is the fourteenth boundary it happened to find. A case cannot be set up from
either: the boundary conditions differ per vessel, one inflow waveform per systemic vein
and a resistance per pulmonary branch, so somebody has to say which cap is the azygous
vein. This is that mapping, validated.

The names become the `mesh-surfaces/` file names and through them the solver's `Add_face`
and `Add_BC` names, so they are what results come back labelled with.

Convention `mesh_complete` relies on: `cap_*` is a face flow crosses, one boundary
condition each; `wall` or `wall_*` is vessel wall, merged into `walls_combined.vtp`.

Where the names are *kept* is the host's business, not this module's. The Mesh Prep panel
keeps them in the scene it was named in.
"""

from __future__ import annotations

from dataclasses import dataclass

CAP_PREFIX = "cap_"
WALL_PREFIX = "wall_"


class FaceTableError(ValueError):
    """Raised when a face table is missing, malformed or inconsistent."""


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
        return self.name == "wall" or self.name.startswith(WALL_PREFIX)


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
