# SimVascular Mesh Prep

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

**Mesh** — the volume mesh node, and which cell array its face ids are in. Four buttons
beside the selector, for seeing a cap that sits inside the anatomy: **edges** on or off,
**colour by face ids** so every face can be told from its neighbours at once (turning that
off leaves the mesh a slightly see-through neutral grey rather than whatever colour its
node was created with), **transparency**, and **show or hide**. None of them is checkable — a mark would be saying what
the display node holds, and nothing tells the panel when that changes elsewhere, so it
would sooner or later contradict the scene. Each reads the state at the moment it is
pressed and turns it around. *Face ids
array* takes several names and uses the first the mesh carries: `CellEntityIds` is VMTK's
and `ModelFaceID` is SimVascular's. A mesh whose ids went somewhere else needs that name
here — CFD Mesh Generator writes them under whichever name its own field lists first that
the *input surface* already carried, which for a surface out of Clip Vessel can be
`MaterialIds`.

**Faces** — a row per face, with its cell count, area, the diameter of the circle of the
same area, and its flatness. Selecting a row shows that face on its own in the 3D view.
The `Name` column is the one to fill in — where it is not already filled in for you: a mesh
that came through Clip Vessel arrives with the names from its clip points, shown dimmed and
italic (see *Names inherited from the clip*).

It works the other way round as well: **moving the cursor over the mesh in a 3D view shows
whichever face is under it**, and **clicking selects that face's row**, scrolling the table
to it if it is out of sight. Which is the direction the work actually goes in — a cap is
something you are looking at before it is a row in a table.

Hovering only shows; it leaves the selection alone, so the highlight goes back to the
selected row when the cursor leaves the mesh. A click at the end of a camera drag is
ignored, so rotating the view does not move the selection. The face under the cursor is
found by where the pick landed rather than by the cell id it also reports: that id indexes
the polydata the display pipeline built to draw the node, which for an unstructured grid
is not the grid's own cells.

**The names you type are saved with the scene**, in the module's parameter node, along with
the face ids array and the output folder. Closing Slicer and opening the scene again brings
them back with the mesh they belong to, and a remesh in the same scene keeps them: they are
matched to faces by id, so only the measurements change. That is the whole of how the
naming persists — there is no file beside the mesh to keep in step with it, and anything
outside Slicer that needs the names reads them out of the scene. Inherited names are not
saved; they are read from the clip points again on every load, so that they cannot disagree
with the clip they came from.

**Export** — writes the folder. The button stays disabled until every face has a name and
there is somewhere to write to, because a face without a name has no boundary condition to
bind to. The folder defaults to `mesh` beside the scene file, which is where a case wants
it, and to nothing at all for a scene that has never been saved.

The line under the panel says how many faces there are, how many came named from Clip
Vessel, and how many are still to name, and follows every name typed. Anything that had to
be done to an inherited name — a duplicate label numbered apart, a clip point that is gone —
is reported there too.

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

## Names inherited from the clip

Most faces should arrive named. The vessel names already exist upstream — they are the
labels on the clip points in [Clip Vessel](https://github.com/vmtk/SlicerExtension-VMTK) —
and a name typed here is the same name typed a second time. So Clip Vessel records on its
output which of its clip points named each face, CFD Mesh Generator carries that record onto
the volume mesh, and this panel follows it: a Fontan case with twenty-six faces opens named
rather than empty.

What travels is three things on the mesh node: a `ClipPoints` node reference to the markups
node, and the attributes `ClipVessel.FaceIdToClipPointID` and `ClipVessel.WallFaceID`. Only
the pointer, never a copy of the labels. That is the whole design: with no second copy there
is nothing to go stale, so renaming a clip point renames the face, live.

A control point label is free text and a face name is a file name, so the labels are put
through the same sanitizer the headless scripts use — `Outlet 1` becomes `cap_Outlet_1` —
and a case packaged from the panel and one packaged from a terminal name their files
identically. Two ends carrying the same label upstream, which Clip Vessel's positional
defaults make easy, are numbered apart (`cap_Outlet_1_3`, `cap_Outlet_1_7`) and reported in
the status line rather than refused: refusing would block an export on names nobody typed,
with nothing to fix in this panel.

### Overrides

**What you type is an override, not the name.** The panel resolves each face as *override,
or else inherited*, which buys three behaviours from one rule:

- rename a clip point upstream and the face name follows;
- type a name and it sticks, whatever happens upstream afterwards;
- clear the cell and the inherited name comes back, rather than the face going blank.

Only the overrides are saved with the scene. The inherited names are worked out again on
every load, from the clip points as they stand then.

**Inherited names are drawn dimmed and italic**, with a tooltip naming the clip point they
came from. This matters more than it sounds. The risk of a name you did not choose is that
it looks exactly like one you did: `cap_Outlet_1`, read off Clip Vessel's positional
default, is indistinguishable from a considered name, and a boundary condition bound to it
is bound to whichever vessel happened to be fourteenth. Seeing which of twenty-six names
nobody has checked is what the panel is for.

### When an inherited name is on the wrong vessel

An inherited name is only as good as the record it came from, and a wrong record is the one fault
nothing else in the chain can see: the face ids are all on the mesh, each names a real clip point,
every name is unique and plausible. It has happened — CFD Mesh Generator's boundary layer once
rotated the cap ids, and on a clinical case 22 of 23 caps were named after another vessel with
nothing anywhere saying so.

So the panel asks the question ids cannot answer: **is this cap where its clip point is?** Each
face's centroid is matched to the clip point positions, one to one so that two faces cannot both
be attributed to the same end and hide a swap between them. A face that comes out somewhere other
than where the record puts it is **drawn in red**, says so when hovered, and is counted in the
status line. Typing a name over it clears the complaint — an override is your own answer, and the
record no longer applies to that face.

It is advisory, not a refusal. A flow extension puts a cap at the tip of the extension rather than
at its clip point, which on a short branch in a crowded tree can read as a disagreement when
nothing is wrong. The tooltip says so. Being wrong the other way costs a boundary condition on
the wrong vessel, which is worth a false alarm or two.

A face whose clip point has been deleted comes out **unnamed**, not renamed. This is why the
record is keyed by control point ID rather than by position in the list: a cap's face id is
`firstCapFaceId + clip point index`, so deleting a clip point shifts every later index down
one, and keyed by index each of those vessels' names would move quietly onto the face of the
vessel before it. Unnamed is a thing an operator can see; a plausible wrong name is not.

Nothing here is required. A mesh imported from elsewhere carries none of the record, and
then every name is typed in this panel, exactly as before.

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

That includes the naming rule. `face_table.names_from_labels` takes `{face_id: label}` and
returns the names, sanitized and with duplicates numbered apart, so a script that has labels
from anywhere — a clip points `.mrk.json`, a spreadsheet — reaches the same file names the
panel would. Only *following Clip Vessel's record* is the module's own, because that means
reading a node reference, which is MRML.
