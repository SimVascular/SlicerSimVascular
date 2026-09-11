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

The table is read from and written to a `face_table.csv` beside the mesh, so a case named
here can be packaged from a script, and a case named in a script can be checked here.
Re-reading a table after a remesh keeps the names and remeasures the faces.

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

## Face ids array

Which cell array the face ids are read from, by the same rule CFD Mesh Generator uses: the
first of the names offered that the mesh carries. `CellEntityIds` is VMTK's name and
`ModelFaceID` is SimVascular's. A mesh whose ids went somewhere else — CFD Mesh Generator
writes them under whichever name its own field lists first that the *input surface* already
carried, which for a surface out of Clip Vessel can be `MaterialIds` — needs that name
given here.

## Outside Slicer

All of it is in `svmeshcomplete`, the package beside this module, which imports nothing
from Slicer and is pip-installable on its own. This file is an MRML adapter: it reads the
selected mesh node's grid, calls the package, and reports what it wrote. A workflow
packaging cases from a terminal calls the same functions and gets the same folder.
