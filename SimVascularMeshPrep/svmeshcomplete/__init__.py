"""The `mesh-complete` folder svMultiPhysics reads, out of a labelled volume mesh.

A mesher hands back one grid holding the volume elements and the boundary cells they
stand on, every cell carrying a face id -- VMTK's convention, and what CFD Mesh Generator
writes. svMultiPhysics reads a folder instead: the volume elements alone under
`GlobalNodeID`/`GlobalElementID`, the exterior surface under `ModelFaceID`, and one file
per named face, which is what its boundary conditions bind to. This is that translation,
and the checks that stop it producing a mesh the solver misreads in silence.

Three things it will not do quietly:

- A volume mesh of more than one element type. The solver decides a mesh's element type by
  counting cell types and taking the last kind found, so tetrahedra with boundary-layer
  prisms among them are read as all prisms, six nodes to an element, and it runs anyway.
- A face table that does not describe the mesh. A cap the mesher lost is a hole in the
  domain; a cap renumbered since the table was written is a boundary condition on the
  wrong vessel.
- A boundary cell standing against no volume element, which the solver has nothing to
  integrate the boundary term against.

`faces.measure_faces` is the other half: area, effective diameter, centroid and flatness
per face, which is how a cap gets matched to a vessel and how a cut that was not planar
is spotted. `face_table.FaceTable` is the naming those measurements inform -- the names
become the file names, and through them the solver's `Add_face` and `Add_BC` names. Where
a host *keeps* those names is its own business; this only validates them.

Nothing here imports `slicer`, `sv` or VMTK: it is VTK and numpy, so it runs under
Slicer's Python and outside it, and its tests need neither.
"""

from svmeshcomplete import testing
from svmeshcomplete.face_table import (
    CAP_PREFIX,
    WALL_NAME,
    WALL_PREFIX,
    DerivedNames,
    Face,
    FaceTable,
    FaceTableError,
    cap_name,
    names_from_labels,
    sanitized,
)
from svmeshcomplete.faces import (
    FACE_ID_ARRAY_NAMES,
    FaceGeometry,
    FaceGeometryError,
    boundary_of,
    find_face_id_array,
    measure_faces,
)
from svmeshcomplete.io import read_dataset, write_dataset
from svmeshcomplete.mesh_complete import (
    EXTERIOR_SURFACE_NAME,
    MESH_SURFACES_DIR_NAME,
    VOLUME_MESH_NAME,
    WALLS_COMBINED_NAME,
    MeshCompleteError,
    MeshCompleteResult,
    write_mesh_complete,
)

__all__ = [
    "testing",
    "CAP_PREFIX",
    "DerivedNames",
    "EXTERIOR_SURFACE_NAME",
    "FACE_ID_ARRAY_NAMES",
    "Face",
    "FaceGeometry",
    "FaceGeometryError",
    "FaceTable",
    "FaceTableError",
    "MESH_SURFACES_DIR_NAME",
    "MeshCompleteError",
    "MeshCompleteResult",
    "VOLUME_MESH_NAME",
    "WALLS_COMBINED_NAME",
    "WALL_NAME",
    "WALL_PREFIX",
    "boundary_of",
    "cap_name",
    "find_face_id_array",
    "measure_faces",
    "names_from_labels",
    "read_dataset",
    "sanitized",
    "write_dataset",
    "write_mesh_complete",
]
