"""Cell-level VTK helpers shared by the geometry modules."""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# vtkCellTypes' static helpers moved to vtkCellTypeUtilities in VTK 9.6, where the old
# ones warn on every call.
CELL_TYPES = getattr(vtk, "vtkCellTypeUtilities", vtk.vtkCellTypes)


def dimensions(dataset):
    """The topological dimension of every cell, in order."""
    return np.array(
        [
            CELL_TYPES.GetDimension(dataset.GetCellType(index))
            for index in range(dataset.GetNumberOfCells())
        ],
        dtype=np.int64,
    )


def extract(dataset, cell_indices):
    """The named cells, with only the points they use."""
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    ids.SetNumberOfTuples(len(cell_indices))
    for position, cell_index in enumerate(cell_indices):
        ids.SetTuple1(position, int(cell_index))

    node = vtk.vtkSelectionNode()
    node.SetFieldType(vtk.vtkSelectionNode.CELL)
    node.SetContentType(vtk.vtkSelectionNode.INDICES)
    node.SetSelectionList(ids)
    selection = vtk.vtkSelection()
    selection.AddNode(node)

    extract_filter = vtk.vtkExtractSelection()
    extract_filter.SetInputData(0, dataset)
    extract_filter.SetInputData(1, selection)
    extract_filter.PreserveTopologyOff()
    extract_filter.Update()
    return extract_filter.GetOutput()


def as_polydata(dataset):
    if isinstance(dataset, vtk.vtkPolyData):
        return dataset
    surface = vtk.vtkGeometryFilter()
    surface.SetInputData(dataset)
    surface.MergingOff()
    surface.PassThroughCellIdsOff()
    surface.PassThroughPointIdsOff()
    surface.Update()
    return surface.GetOutput()


def add_int_array(dataset, name: str, values, *, on_points: bool) -> None:
    array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.int32), deep=True)
    array.SetName(name)
    target = dataset.GetPointData() if on_points else dataset.GetCellData()
    target.AddArray(array)


def keep_only(attributes, names) -> None:
    for index in reversed(range(attributes.GetNumberOfArrays())):
        if attributes.GetArrayName(index) not in names:
            attributes.RemoveArray(index)


def element_type_name(dataset) -> str:
    """The one cell type of a dataset, or raise if it has more than one."""
    types = {dataset.GetCellType(index) for index in range(dataset.GetNumberOfCells())}
    if len(types) != 1:
        names = ", ".join(sorted(CELL_TYPES.GetClassNameFromTypeId(kind) for kind in types))
        raise ValueError(f"Expected one cell type, found {names}")
    return CELL_TYPES.GetClassNameFromTypeId(types.pop()).replace("vtk", "")
