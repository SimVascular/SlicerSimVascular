"""Which face id is which vessel, and what the solver calls it.

Clip Vessel labels the faces it cuts `Wall`, `Inlet`, `Outlet 1` ... `Outlet 22` and
writes that table beside the model as `Clip Vessel face colors.csv`. Those names are
positional, so a Fontan case cannot be set up from them: the operator has to say which
cap is the azygous vein and which is `lpa_f`. That mapping is `geometry/face_table.csv`,
and the names in it become the `mesh-surfaces/` file names and through them the
`Add_face` and `Add_BC` names in solver.xml.

Convention `mesh_complete` and `solver_xml` both rely on: `cap_*` is a face flow crosses,
one boundary condition each; `wall` or `wall_*` is vessel wall, merged into
`walls_combined.vtp`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

CAP_PREFIX = "cap_"
WALL_PREFIX = "wall_"
FIELDNAMES = ("FaceID", "ClipVesselName", "Name")


class FaceTableError(ValueError):
    """Raised when a face table is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Face:
    face_id: int
    name: str
    clip_vessel_name: str = ""

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

    @classmethod
    def read(cls, path: str | Path) -> "FaceTable":
        path = Path(path)
        if not path.is_file():
            raise FaceTableError(f"No face table at {path}")
        faces = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"FaceID", "Name"} - set(reader.fieldnames or ())
            if missing:
                raise FaceTableError(
                    f"{path} has no {sorted(missing)} column; it needs {list(FIELDNAMES)}"
                )
            for line, row in enumerate(reader, start=2):
                raw_id = (row.get("FaceID") or "").strip()
                name = (row.get("Name") or "").strip()
                if not raw_id:
                    continue
                if not name:
                    raise FaceTableError(
                        f"{path} line {line}: face {raw_id} has no name. Naming every "
                        "face is what this file is for."
                    )
                try:
                    face_id = int(raw_id)
                except ValueError:
                    raise FaceTableError(
                        f"{path} line {line}: FaceID {raw_id!r} is not an integer"
                    ) from None
                faces.append(
                    Face(face_id, name, (row.get("ClipVesselName") or "").strip())
                )
        if not faces:
            raise FaceTableError(f"{path} names no faces")
        return cls(faces)

    def write(self, path: str | Path) -> Path:
        return _write_rows(
            path,
            [
                {
                    "FaceID": str(face.face_id),
                    "ClipVesselName": face.clip_vessel_name,
                    "Name": face.name,
                }
                for face in self.faces
            ],
        )


def read_clip_vessel_table(path: str | Path) -> dict[int, str]:
    """Clip Vessel's colour table (`LabelValue,Name,Color_R,...`) as `{face id: label}`."""
    path = Path(path)
    if not path.is_file():
        raise FaceTableError(f"No Clip Vessel face table at {path}")
    labels: dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"LabelValue", "Name"} <= set(reader.fieldnames or ()):
            raise FaceTableError(
                f"{path} has no LabelValue/Name columns, so it is not the Slicer colour "
                "table Clip Vessel writes as 'Clip Vessel face colors.csv'."
            )
        for row in reader:
            raw_id = (row.get("LabelValue") or "").strip()
            label = (row.get("Name") or "").strip()
            if raw_id and label:
                labels[int(raw_id)] = label
    if not labels:
        raise FaceTableError(f"{path} names no faces")
    return labels


def starter_rows(clip_vessel_labels: dict[int, str]) -> list[dict[str, str]]:
    """Rows for a `face_table.csv` to be filled in by hand.

    A wall is named outright; every other face is guessed at as `cap_<label>`, which is a
    placeholder -- `cap_outlet_14` is a name no result should be read under.
    """
    rows = []
    for face_id in sorted(clip_vessel_labels):
        label = clip_vessel_labels[face_id]
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
        # `Outlet 2` -> `outlet_02`: face names get sorted, and unpadded numbers sort
        # `outlet_14` before `outlet_2`.
        trailing_number = re.fullmatch(r"(.*?)_(\d+)", slug)
        if trailing_number:
            slug = f"{trailing_number.group(1)}_{int(trailing_number.group(2)):02d}"
        if slug == "wall":
            name = "wall"
        elif slug.startswith("wall"):
            name = f"{WALL_PREFIX}{slug[4:].strip('_')}"
        else:
            name = f"{CAP_PREFIX}{slug}"
        rows.append({"FaceID": str(face_id), "ClipVesselName": label, "Name": name})
    return rows


def write_starter_table(clip_vessel_labels: dict[int, str], path: str | Path) -> Path:
    return _write_rows(path, starter_rows(clip_vessel_labels))


def _write_rows(path: str | Path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path
