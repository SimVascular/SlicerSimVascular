"""svReducedOrderModelingUtils' testing aorta: fetched, checked, and read as a case is.

`case01` of the package's
[TESTING_DATA](https://github.com/SimVascular/svReducedOrderModelingUtils/releases/tag/TESTING_DATA)
release is the model the whole-case tests run on, here and in the module's own tests under
Slicer. It is a real aorta's SimVascular mesh-complete -- an aortic valve, an arch with four
branches, the descending aorta -- with the inflow, `rcrt.dat` and `resistance.dat` its tutorial
gives it, and the package's own centerlines of it. A model from elsewhere rather than one made
here, so that the tests hold a case to something they did not compute themselves: the centerlines
to the release's, each outlet to its file's values.

Downloaded rather than kept in this repository, at 17 MB, and checked against the SHA-256 digests
GitHub publishes for the release's files. A download cut short is fetched again; a file that is
not the one the tests were written against -- the release changed since, or a folder of someone
else's -- is refused by name rather than tested, since what the tests expect was taken from these.

Part of the package rather than of its tests, as `svmeshcomplete.testing` is, because the module's
tests under Slicer want the same aorta. Nothing here imports Slicer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from svmeshcomplete import io
from svmeshcomplete.mesh_complete import EXTERIOR_SURFACE_NAME, MESH_SURFACES_DIR_NAME

from svromsetup import boundary_conditions as bcs

RELEASE_URL = "https://github.com/SimVascular/svReducedOrderModelingUtils/releases/download/TESTING_DATA"
# The release's files the tests read, by the SHA-256 GitHub gives for each.
FILES = {
    "case01_mesh.zip": "6358439aee645d09e91ebb34a6e7ca1f4506de8ad29b1c964510c79579ecc71b",
    "case01_centerlines.vtp": "a2256c024bcdf72399d44039aa717f777da27704c3be636933ffc16b68fa6d97",
    "case01_inflow.flow": "69199991dface483e50058d70d61052fa27d7aaa672eba34948dc42156e7edf6",
    "rcrt.dat": "570270a39087ce3304ef81f8d182e6535062f04491bb8c5638c1b0cc8a6e320f",
    "resistance.dat": "b6120802fce791757f1a0147eeba988804329f6e520935192835a0baafc082fe",
}
MESH_ZIP = "case01_mesh.zip"
MESH_DIR_NAME = "mesh"
# A folder holding the release's files as it has them, for a machine that cannot download them.
FOLDER_VARIABLE = "SVROM_TESTING_DATA"

INLET = "cap_aortic_valve"
RCR_OUTLET = "cap_distal_descending_aorta"
FACE_ID_ARRAY = "ModelFaceID"


class TestingDataUnavailable(RuntimeError):
    """The data could not be had -- offline, or a folder without it -- so the tests that need it
    are skipped, and say why."""


class TestingDataChanged(RuntimeError):
    """A file that is not the one the tests were written against. Not a reason to skip: a test of
    other data against these expectations would fail, or worse, pass, for reasons of its own."""


@dataclass(frozen=True)
class Case01:
    folder: Path
    """Where the release's files are."""
    mesh_dir: Path
    """The mesh-complete, unpacked: the volume mesh, the exterior surface, `mesh-surfaces/`."""
    names: dict
    """`{face id: name}`, by the mesh-surfaces files, as SimVascular names faces."""
    centres: dict
    """`{name: centre}`, each face's points' mean, in the mesh's millimetres."""
    conditions: dict
    """`{name: condition}`: the inflow on the valve, and each outlet's from its `.dat` file."""

    @property
    def exterior_path(self) -> Path:
        return self.mesh_dir / EXTERIOR_SURFACE_NAME

    @property
    def inflow_path(self) -> Path:
        return self.folder / "case01_inflow.flow"

    @property
    def centerlines_path(self) -> Path:
        return self.folder / "case01_centerlines.vtp"

    def face_id(self, name: str) -> int:
        return next(face_id for face_id, face_name in self.names.items() if face_name == name)

    def surface(self):
        """The exterior surface, its faces under `ModelFaceID`: a labelled surface, which is all a 0D
        model needs of the geometry. Read again for each caller, so that each has its own."""
        return io.read_dataset(self.exterior_path)


def fetch(folder, download: bool = True) -> Path:
    """`folder`, holding every file the tests read, each checked; those missing downloaded into it
    when `download`, and refused otherwise."""
    folder = Path(folder)
    if download:
        folder.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = folder / name
        if not path.is_file():
            if not download:
                raise TestingDataUnavailable(f"{folder} has no {name}, which the release has.")
            _download(name, path)
        found = _sha256(path)
        if found != digest:
            raise TestingDataChanged(
                f"{path} is not the release's {name} the tests were written against: its SHA-256 is "
                f"{found}, not {digest}.")
    return folder


def unpack(folder, into) -> Path:
    """The mesh folder, `case01_mesh.zip` unpacked into `into` -- once: one already there is used.

    Unpacked beside it and then moved in whole, so that one cut short is never read as the mesh;
    and refused when the archive holds a path outside the folder it unpacks into.
    """
    into = Path(into)
    mesh = into / MESH_DIR_NAME
    if (mesh / EXTERIOR_SURFACE_NAME).is_file() and (mesh / MESH_SURFACES_DIR_NAME).is_dir():
        return mesh
    into.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=into, prefix=".unpacking-"))
    try:
        with zipfile.ZipFile(Path(folder) / MESH_ZIP) as archive:
            for member in archive.namelist():
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise TestingDataChanged(f"{MESH_ZIP} holds {member!r}, outside the folder it unpacks into.")
            archive.extractall(staging)
        if mesh.exists():
            shutil.rmtree(mesh)
        (staging / MESH_DIR_NAME).rename(mesh)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return mesh


def read_case01(folder, mesh_dir) -> Case01:
    """The aorta as a case reads it: its faces named by their files, its conditions by theirs."""
    folder, mesh_dir = Path(folder), Path(mesh_dir)
    names, centres = {}, {}
    for path in sorted((mesh_dir / MESH_SURFACES_DIR_NAME).glob("*.vtp")):
        face = io.read_dataset(path)
        (face_id,) = np.unique(vtk_to_numpy(face.GetCellData().GetArray(FACE_ID_ARRAY)))
        names[int(face_id)] = path.stem
        centres[path.stem] = vtk_to_numpy(face.GetPoints().GetData()).mean(axis=0)
    conditions = {INLET: bcs.read_flow_file(folder / "case01_inflow.flow"),
                  **bcs.read_rcrt(folder / "rcrt.dat"), **bcs.read_resistance(folder / "resistance.dat")}
    return Case01(folder, mesh_dir, names, centres, conditions)


def case01(cache, scratch=None) -> Case01:
    """The aorta: from the folder `SVROM_TESTING_DATA` names, as it is, or else from `cache`,
    downloaded into it the first time.

    Unpacked into `cache`, so that it is unpacked once; a named folder may be someone's own, and
    nothing is written into it, so it is unpacked into `scratch`, a new temporary folder unless one
    is given.
    """
    named = os.environ.get(FOLDER_VARIABLE)
    if named:
        folder = fetch(named, download=False)
        into = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="case01-"))
    else:
        folder = into = fetch(cache)
    return read_case01(folder, unpack(folder, into))


def release_centerlines(path, inlet_centre):
    """The release's centerlines, one per outlet, as `(end, length)`: where each ends, farthest from
    the valve, and how long it is, in the mesh's millimetres. Measured along its points, since its
    `Path` starts again at every branch."""
    lines = io.read_dataset(path)
    points = vtk_to_numpy(lines.GetPoints().GetData())
    member = vtk_to_numpy(lines.GetPointData().GetArray("CenterlineId")).astype(bool)
    lengths = np.zeros(member.shape[1])
    for index in range(lines.GetNumberOfCells()):
        cell = lines.GetCell(index)
        chain = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        for first, second in zip(chain, chain[1:]):
            lengths[member[first] & member[second]] += np.linalg.norm(points[first] - points[second])
    found = []
    for line in range(member.shape[1]):
        on = points[member[:, line]]
        found.append((on[np.argmax(np.linalg.norm(on - inlet_centre, axis=1))], lengths[line]))
    return found


def _download(name: str, path: Path) -> None:
    partial = path.with_name(name + ".part")
    try:
        with urllib.request.urlopen(f"{RELEASE_URL}/{name}", timeout=60) as response:
            partial.write_bytes(response.read())
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise TestingDataUnavailable(
            f"svReducedOrderModelingUtils' testing data could not be downloaded ({error}); set "
            f"{FOLDER_VARIABLE} to a folder holding {', '.join(FILES)}.") from None
    # Renamed once whole, so that a download cut short is fetched again rather than read.
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
