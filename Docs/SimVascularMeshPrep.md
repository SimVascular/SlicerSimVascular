# Mesh Complete Export

Turns a face-labelled volume mesh into the `mesh-complete` folder an svMultiPhysics case
reads, and lets you name the faces on the way. The input is a volume mesh — typically one
out of [CFD Mesh Generator](CfdMeshGenerator.md), whose face ids this reads — and the
output is a folder of files the solver binds its boundary conditions through.

## What the solver wants, and why this is a step at all

A mesher hands back one grid holding the volume elements and the boundary cells they stand
on, every cell carrying a face id. svMultiPhysics reads a folder:

```
mesh/
  mesh-complete.mesh.vtu        volume elements alone, GlobalNodeID + GlobalElementID
  mesh-complete.exterior.vtp    every boundary cell, with ModelFaceID
  walls_combined.vtp            the wall faces merged, for the no-slip condition
  mesh-surfaces/cap_RSVC.vtp    one file per named face
```

`GlobalNodeID` is how a boundary condition is bound to the volume: each face file carries,
per point, the id of the volume node it is, and the solver looks each one up.
`GlobalElementID` on a face is the element behind each boundary cell, which is what the
boundary term is integrated against. No mesher writes any of this, which is why the
translation is a step of its own.

## The panel

**Mesh** — the volume mesh node, and which cell array its face ids are in. *Face ids
array* takes several names and uses the first the mesh carries: `CellEntityIds` is VMTK's
and `ModelFaceID` is SimVascular's. A mesh whose ids went somewhere else needs that name
here — CFD Mesh Generator writes them under whichever name its own field lists first that
the *input surface* already carried, which for a surface out of Clip Vessel can be
`MaterialIds`.

**Faces** — a row per face, with its cell count, area, the diameter of the circle of the
same area, and its flatness. Selecting a row shows that face on its own in the 3D view,
and *Hide the mesh* takes the rest of the anatomy away, which is the only way to see a cap
that sits inside it. The `Name` column is the one to fill in.

**Face table file** — the same `face_table.csv` the command line tools read and write, so
a case named here can be packaged by a script and one named by a script can be checked
here. Loading a table keeps its names and remeasures the faces against the mesh now
selected, which is what makes a remesh cheap: the naming survives it.

**Export** — writes the folder. The button stays disabled until every face has a name,
because a face without one has no boundary condition to bind to.

## Naming the faces

A face id is a number; a boundary condition is per vessel. The panel lists every face of
the mesh with its area, the diameter of the circle of the same area, its centroid and its
flatness, highlights the selected one in the 3D view, and takes a name for it.

The names matter beyond this module: they become the `mesh-surfaces/` file names, and
through those the `Add_face` and `Add_BC` names in `solver.xml` — so they are what every
result comes back labelled with. `cap_RSVC` is a name to read a pressure under;
`cap_14` is not.

Convention: `cap_*` for a face the flow crosses, one boundary condition each, and `wall`
or `wall_*` for vessel wall, which is what gets merged into `walls_combined.vtp`.

**Flatness is the check on a cap.** It is the largest distance of any point of the face
from its own best-fit plane, so a cap cut normal to the vessel reads 0. One that does not
was not cut cleanly, and the flow crossing it is not what its boundary condition says.

## What it refuses

Three things, because all three otherwise reach the solver and are misread in silence
rather than reported:

- **A volume mesh of more than one element type.** svMultiPhysics decides a mesh's element
  type by counting cell types and taking the last kind it found, so tetrahedra with
  boundary-layer prisms among them are read as all prisms, six nodes to an element. In CFD
  Mesh Generator, *Tetrahedralize* is what prevents this.
- **A face table that does not describe the mesh.** A cap the mesher lost is a hole in the
  domain; a cap renumbered since the table was written is a boundary condition on the
  wrong vessel.
- **A boundary cell standing against no volume element**, which the solver has nothing to
  integrate the boundary term against.

## Against SimVascular's own writer

The format is SimVascular's, so `svmeshcomplete` follows
[`sv4gui_MeshLegacyIO.cxx`](https://github.com/SimVascular/SimVascular/blob/master/Code/Source/sv4gui/Modules/Mesh/Common/sv4gui_MeshLegacyIO.cxx)'s
`WriteFiles` rather than inventing a compatible-looking format of its own.

The one semantic worth stating is the ids. SimVascular's `ResetFaceSurfaceIds` rewrites a
face's `GlobalNodeID` to `node_map[id] + 1` and its `GlobalElementID` to
`elem_map[id] + 1`, where those maps take an id to its *index* in the volume mesh's
arrays — so a face's ids are the 1-based positions of its nodes and its owning elements in
the volume mesh, which is what the solver looks them up as. SimVascular has to remap
because the mesher hands it ids it did not choose; here the ids are assigned over the
volume mesh's own points and cells, so the same invariant holds by construction.

Otherwise: faces are extracted by thresholding `ModelFaceID` with the points compacted,
as `PlyDtaUtils_GetFacePolyData` does; everything is written zlib-compressed, appended and
un-encoded, as SimVascular's writers are; and `walls_combined.vtp` is the wall faces
together. SimVascular appends its separately extracted faces and cleans them with point
merging, because extracting them separately duplicated the points along every seam; here
the wall cells come out of the exterior in one pass, so there is nothing to merge.

Two deliberate differences:

- **A wall in more than one piece.** SimVascular writes
  `walls_combined_connected_region_<j>.vtp` per piece *instead of* `walls_combined.vtp`.
  This writes those files too, and `walls_combined.vtp` as well, because a case config
  naming a file the export decided not to write fails at the solver rather than here. The
  panel says how many pieces there were, which is the part worth acting on.
- **More than one `ModelRegionID`.** SimVascular splits a multi-domain mesh into
  `<dir>_domain-<i>` folders. This refuses instead, rather than flattening the regions
  into one without saying so. A mesh carrying a single region keeps its id.

## Face ids array

Which cell array the face ids are read from, by the same rule CFD Mesh Generator uses: the
first of the names offered that the mesh carries. `CellEntityIds` is VMTK's name,
`ModelFaceID` is SimVascular's, and `MaterialIds` is Slicer's own material array.

That last one is in the default list, and last in it, because CFD Mesh Generator puts the
ids there: its own field uses the first name the *input surface* already carried, and its
widget prepends an integer cell array it finds on an input carrying neither of the others
— which a surface out of Clip Vessel is. A mesh built on one arrives labelled by
`MaterialIds` and is no less labelled for it, so there is nothing to set and nothing to
redo upstream.

Reading it cannot quietly mistake a real material array for face labels. A genuine one has
nonzero values on the volume elements and a face id array does not, so the export refuses
it and says so. Being last means a mesh carrying both is read by the name that means
faces.

Whichever array the input used, the exported files carry `ModelFaceID`, so the choice does
not reach anything downstream.

## Outside Slicer

All of it is in `svmeshcomplete`, the package beside this module, which imports nothing
from Slicer and is pip-installable on its own. This file is an MRML adapter: it reads the
selected mesh node's grid, calls the package, and reports what it wrote. A workflow
packaging cases from a terminal calls the same functions and gets the same folder.
