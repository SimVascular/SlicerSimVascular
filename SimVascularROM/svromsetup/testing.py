"""A vessel to set a 0D model up on, without a clinical mesh, a mesher or Slicer.

A Y: an inlet arm down the z axis, two outlet arms up and out in the xz plane, each cut square
and capped, face 1 the wall, 2 the inlet, 3 the outlet to +x and 4 the outlet to -x. It is the
smallest geometry that has a bifurcation, which is what the centerlines are split at and what
makes the outlets' order a question at all.

The arms are of different radii on purpose. Two outlets the same size would come out of the
model with the same resistance and the same flow, and a test that their conditions landed on
the right ones could not then tell them apart.

Part of the package rather than of its tests for the reason `svmeshcomplete.testing` is: the
host's own tests want the same vessel.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

WALL_ID, INLET_ID, RIGHT_ID, LEFT_ID = 1, 2, 3, 4
NAMES = {WALL_ID: "wall", INLET_ID: "cap_inlet", RIGHT_ID: "cap_right", LEFT_ID: "cap_left"}


def y_arms(length: float = 30.0, angle_degrees: float = 40.0):
    """Each arm's direction, radius and cut, by the face id of its cap."""
    angle = np.radians(angle_degrees)
    return {
        INLET_ID: (np.array([0.0, 0.0, -1.0]), 3.0),
        RIGHT_ID: (np.array([np.sin(angle), 0.0, np.cos(angle)]), 2.4),
        LEFT_ID: (np.array([-np.sin(angle), 0.0, np.cos(angle)]), 1.8),
    }, length


def y_surface(length: float = 30.0, spacing: float = 0.4):
    """The capped Y as a triangulated surface carrying `ModelFaceID`, in millimetres."""
    arms, length = y_arms(length)
    extent = length + 6.0
    dims = int(2 * extent / spacing) + 1
    # Off the cuts by a fraction of a voxel: a grid node exactly on a cut is a zero of the field,
    # which the contour turns into slivers.
    origin = -extent + 0.37 * spacing
    axis = origin + spacing * np.arange(dims)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([x, y, z], axis=-1)

    # Signed distance to the union of three arms, each a tube from the origin cut square at its
    # end: the larger of the distance to the tube and the distance past the cut. Cut in the field
    # rather than clipped afterwards, so the caps come out of the contour flat and already joined
    # to the wall. The second term is linear, so where it decides the surface the contour's own
    # interpolation puts every point exactly in the cut.
    distance = np.full(x.shape, np.inf)
    for direction, radius in arms.values():
        along = points @ direction
        nearest = np.clip(along, 0.0, None)[..., None] * direction
        tube = np.linalg.norm(points - nearest, axis=-1) - radius
        distance = np.minimum(distance, np.maximum(tube, along - length))

    image = vtk.vtkImageData()
    image.SetDimensions(dims, dims, dims)
    image.SetSpacing(spacing, spacing, spacing)
    image.SetOrigin(origin, origin, origin)
    scalars = numpy_to_vtk(distance.ravel(), deep=True)
    image.GetPointData().SetScalars(scalars)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(contour.GetOutputPort())
    # A collapsed triangle would otherwise come out as a line, and the face ids are per polygon.
    clean.ConvertPolysToLinesOff()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    surface = normals.GetOutput()

    # A cell is on a cap when it lies in that arm's cut and faces along the arm.
    centres = vtk.vtkCellCenters()
    centres.SetInputData(surface)
    centres.Update()
    centre = vtk_to_numpy(centres.GetOutput().GetPoints().GetData())
    normal = vtk_to_numpy(surface.GetCellData().GetNormals())
    face_ids = np.full(surface.GetNumberOfCells(), WALL_ID, dtype=np.int32)
    for face_id, (direction, _radius) in arms.items():
        on_cut = np.abs(centre @ direction - length) < 0.01 * spacing
        facing = np.abs(normal @ direction) > 0.999
        face_ids[on_cut & facing] = face_id

    labelled = vtk.vtkPolyData()
    labelled.SetPoints(surface.GetPoints())
    labelled.SetPolys(surface.GetPolys())
    array = numpy_to_vtk(face_ids, deep=True)
    array.SetName("ModelFaceID")
    labelled.GetCellData().AddArray(array)
    return labelled


def cap_centres(length: float = 30.0):
    """Where each cap's centre is, by face id."""
    arms, length = y_arms(length)
    return {face_id: direction * length for face_id, (direction, _radius) in arms.items()}
