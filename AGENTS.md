# Notes for coding agents

3D Slicer extension holding the SimVascular modules the Marsden lab develops. Four scripted
modules, registered in `CMakeLists.txt`: `SDFStent`, `PaintModel`, `FaceAwareRemesh`,
`SimVascularMeshPrep`. Each has a page under `Docs/`, linked from `README.md`.

Paths below are relative to the repository root. The last section covers optional local
tooling, and says where it is assumed to live.

## The layout that matters

`FaceAwareRemesh` and `SimVascularMeshPrep` each split in two:

- **The scripted module** (`FaceAwareRemesh.py`, `SimVascularMeshPrep.py`) — an MRML adapter.
  It reads the selected node, calls the package, puts the answer in the panel.
- **A package beside it** (`svremesh/`, `svmeshcomplete/`) — the actual geometry, with its own
  `pyproject.toml` and `LICENSE.txt`, pip-installable on its own.

**The packages import no `slicer`, no `sv`, no VMTK — only VTK and numpy.** That is the point of
the split: a workflow packaging cases from a terminal calls the same functions and gets the same
answer, and the tests need neither Slicer nor a mesher. Keep it that way. If a change needs
MRML, it belongs in the module file, not the package.

`SDFStent` and `PaintModel` predate that pattern and don't follow it.

## Running the tests

Both suites are headless: they need only numpy and VTK, no Slicer and no mesher. Any interpreter
with those works — a venv with `pip install numpy vtk pytest` is enough. Note that the Python on
`PATH` on macOS usually has neither.

**svremesh** — 85 tests, unittest, about 12 seconds:

```sh
cd FaceAwareRemesh && PYTHONPATH=. python -m unittest discover -s tests
```

**svmeshcomplete** — pytest, `testpaths = Testing/Python` in its `pyproject.toml`:

```sh
cd SimVascularMeshPrep && python -m pytest
```

Neither suite is wired into CTest yet: the `Testing/CMakeLists.txt` files only recurse, and
`FaceAwareRemesh/Testing/Python/CMakeLists.txt` has its `slicer_add_python_unittest` line
commented out. Run them directly, as above.

`PythonSlicer` (in `Slicer.app/Contents/bin` on macOS) is **not** usable non-interactively — its
launcher swallows `-c` and script arguments and prints its own help instead. The application
binary is another matter; see the next section.

## Testing anything that touches MRML

Panel behaviour, node references, node attributes, scene `.mrb` round-trips — none of it is
reachable from the headless suites, but all of it is scriptable. The application binary takes
`--python-script`, which is how `slicer_add_python_test` runs the suites, so a repo test file
runs unchanged:

```sh
/Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window --testing \
  --additional-module-paths SimVascularMeshPrep \
  --python-script SimVascularMeshPrep/Testing/Python/some_test.py
```

- **Drop `--no-main-window` for widget tests**, and for anything that could disturb the layout
  manager: `vtkMRMLScene.Clear(1)` removes the layout node and segfaults a windowed Slicer, which
  is invisible headless. Use `Clear()` and remove the parameter node by hand instead.
- **Drop `--disable-cli-modules`** for anything that preprocesses a surface, which decimates
  through the `decimation` CLI module.
- `--testing` uses a throwaway settings file, so no module is registered from the user's
  configured paths — pass `--additional-module-paths` for anything that needs
  `slicer.util.getModuleWidget`.
- `slicer.util.getModuleWidget` returns **the same widget every time**. State a test leaves on it
  outlives `mrmlScene.Clear()`, so reset what you rely on in `setUp`.

Prefer this to driving a live Slicer over MCP for anything repeatable: a clean scene every run,
and a crash costs nothing. Running a whole suite through MCP `execute_python` has segfaulted
Slicer and taken the server with it. Keep MCP for looking at a session, not for running tests.

Write it so as much as possible falls on the package side anyway, where it needs no Slicer at all.

## Writing style

The prose here is load-bearing and unusually dense — read `Docs/SimVascularMeshPrep.md` or the
module docstring in `SimVascularMeshPrep.py` before writing any. The conventions:

- **Docstrings and docs say _why_, not what.** A comment restating the code is noise; a comment
  explaining the constraint that forced the code is the whole value. Most non-obvious lines in
  this repo carry a sentence on what goes wrong without them, often naming the failure.
- **Name the consequence.** "The solver decides a mesh's element type by counting cell types and
  taking the last kind it found, so tetrahedra with boundary-layer prisms among them are read as
  all prisms" — not "validate element types".
- **Deliberate decisions get recorded as decisions**, including what was rejected and why. See
  the "Against SimVascular's own writer" section of `Docs/SimVascularMeshPrep.md`.
- British spelling in prose (`labelled`, `colour`) — but **not** in identifiers, which follow
  VTK and Slicer (`SetColor`, `ModelFaceID`, `face_id`).
- Module files are `camelCase` (Slicer convention); the packages beside them are `snake_case`
  (PEP 8). Both are correct in their own half.

Match the surrounding density. A terse patch in this codebase reads as unfinished.

## Upstream this repo depends on

[SlicerExtension-VMTK](https://github.com/vmtk/SlicerExtension-VMTK)'s `ClipVessel` and
`CfdMeshGenerator` produce the face-labelled volume meshes `SimVascularMeshPrep` consumes, so
their face id conventions are this repo's input contract:

- A boundary label and a face id are **one numbering** — the vessel end labelled 2 in
  `BoundaryLabels` point data is face 2 in the face id cell data, in both modules.
- Layout of a clipped surface: pre-existing input faces compacted from 1, then the wall, then
  one cap per clip point. With nothing pre-existing that is wall 1, caps 2, 3, …
- Face ids arrive under `CellEntityIds` (VMTK), `ModelFaceID` (SimVascular) or `MaterialIds`
  (Slicer), depending on what the input surface carried. `Docs/SimVascularMeshPrep.md`
  explains which and why.
- **The face *names* travel too, as a pointer.** Clip Vessel puts a `ClipPoints` node reference
  and the attributes `ClipVessel.FaceIdToClipPointID` / `ClipVessel.WallFaceID` on its output, and
  CFD Mesh Generator copies them onto the volume mesh. The names themselves are control point
  labels on the markups node, never copied, so renaming a clip point renames the face. The map is
  keyed by control point **ID**, not index: a cap's face id is `firstCapFaceId + clip point
  index`, so keying by index would move names onto neighbouring faces when a clip point is
  deleted. `Docs/SimVascularMeshPrep.md` has the whole chain.

Changing anything here that reads face ids means checking those conventions upstream first.

Two things that cost real time when working against a VMTK working tree:

- **Slicer loads `CfdMeshGeneratorLib` from the *installed* SlicerVMTK extension** even when
  `CfdMeshGenerator.py` resolves to a working tree, because the installed `qt-scripted-modules`
  sits first on `sys.path` — and neither `PYTHONPATH` nor `--additional-module-paths` beats it. So
  a session can run worktree module code over installed library code and look fine. Check with
  `from CfdMeshGeneratorLib import MeshingPipeline; print(MeshingPipeline.__file__)`; fix it by
  inserting the worktree at `sys.path[0]`, purging those names from `sys.modules`, and calling
  `slicer.util.reloadScriptedModule`.
- **Ask geometrically, not by id.** A face id being present proves nothing about which vessel it
  is on: a permutation of the cap ids is the same set, all in range, and looks right in the views.
  A boundary layer permuted them for a while and every id-based test passed throughout. Check a
  cap against where its clip point is.

## Optional local tooling

Not required to build, test or contribute. Paths are wherever you cloned things; the committed
`.mcp.json` is gitignored precisely because it holds one machine's absolute paths.

**Driving a live Slicer over MCP.** [pieper/slicer-skill](https://github.com/pieper/slicer-skill)
includes `slicer-mcp-server.py`: paste it into Slicer's Python console and it serves
`list_nodes`, `get_node_properties`, `execute_python`, `screenshot`, `read_file`, `write_file`
and `load_sample_data` on `http://localhost:2026/mcp`. Stop it with `mcpLogic.stop()`. This is
the only practical way to exercise the MRML half of a change.

**Never assume it is running.** It has to be started by hand inside Slicer, and by default the
first request raises a modal dialog asking the user to allow arbitrary Python execution in that
session. Ask.

**Slicer API questions.** The same repository carries `SKILL.md` and a `slicer-skill-search` MCP
server giving ranked search over the Slicer source, extensions and forum archives. If you have
it, **do not clone Slicer repositories into this project** — they belong in that one shared
directory. Its `setup.sh` fetches the corpora; a fresh clone has only some of them.

**The workflow that consumes this extension.** `SimVascular/ClinicalWorkflows` (lab-internal)
holds the Fontan case workflow, whose README is the most concrete account of how these modules
get used end to end, and the best reference for whether a change to Mesh Prep helps or hurts.
It contains no patient data and must stay that way.
