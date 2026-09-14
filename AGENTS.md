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
launcher swallows `-c` and script arguments and prints its own help instead. So there is no way
to test the MRML half of a change from a script; see the next section.

## Testing anything that touches MRML

Panel behaviour, node references, node attributes, scene `.mrb` round-trips — none of it is
reachable from the headless suites, and there is no scripted substitute. It needs a running
Slicer with the extension loaded, driven by hand or over MCP (below).

Write it so as much as possible falls on the package side, where it can be tested.

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

Changing anything here that reads face ids means checking those conventions upstream first.

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
