"""Reading and writing VTK datasets, by file extension.

Kept in one module so that every file this workflow writes is written the same way:
binary, appended, compressed. A mesh folder is a few hundred megabytes as ASCII and a
tenth of that binary, and it gets copied to Sherlock on every run.
"""

from __future__ import annotations

from pathlib import Path

import vtk

_READERS = {
    ".vtu": vtk.vtkXMLUnstructuredGridReader,
    ".vtp": vtk.vtkXMLPolyDataReader,
    ".vtk": vtk.vtkGenericDataObjectReader,
    ".stl": vtk.vtkSTLReader,
}

_WRITERS = {
    ".vtu": vtk.vtkXMLUnstructuredGridWriter,
    ".vtp": vtk.vtkXMLPolyDataWriter,
}


def read_dataset(path: str | Path):
    """Read a VTK dataset, chosen by extension."""
    path = Path(path)
    try:
        reader_class = _READERS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Don't know how to read {path.suffix} files: {path}") from None
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    reader = reader_class()
    reader.SetFileName(str(path))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"Read no points from {path}")
    return dataset


def write_dataset(dataset, path: str | Path) -> Path:
    """Write a VTK dataset as binary, appended and compressed."""
    path = Path(path)
    try:
        writer_class = _WRITERS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Don't know how to write {path.suffix} files: {path}") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = writer_class()
    writer.SetFileName(str(path))
    writer.SetInputData(dataset)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.SetCompressorTypeToZLib()
    writer.Write()
    return path
