# SimVascular ROM Simulation

## Summary

Sets up a 0D (reduced-order) blood flow simulation of a vascular model whose faces are named
in [SimVascular Mesh Prep](SimVascularMeshPrep.md), runs it with
[svZeroDSolver](https://github.com/SimVascular/svZeroDSolver), and shows the results in the
scene. You give one boundary condition per cap and pick the source; the module traces
centerlines from the source with [SlicerVMTK](https://github.com/vmtk/SlicerExtension-VMTK),
builds the 0D model along them with [svromutils](https://pypi.org/project/svromutils/),
writes the svZeroDSolver input file, runs the solver, and reads the CSV back.

A 0D model replaces the three-dimensional flow with a network: each vessel segment becomes a
resistance, an inductance and a capacitance worked out from its length and cross-section. It
solves in well under a second, which makes it the model to tune boundary conditions on, or to
check a 3D case against before running it.

The results come back four ways:
- the pressure over the last cardiac cycle of the caps you select, in Slicer's plot view;
- the centerlines drawn inside the anatomy, coloured by cycle-averaged pressure;
- a CSV of every cap's flow and pressure, exported from the panel;
- the 0D network itself, drawn as a directed graph. Click a block to see its values and
  results: a vessel's are plotted and its stretch of centerline is picked out in 3D.

All four are in Slicer itself, and nothing has to be installed for them.

## Tutorial

1. **Name the faces.** Open the mesh in SimVascular Mesh Prep. A mesh out of Clip Vessel
   arrives named after its clip points, and opening it there is enough to save those names on
   it. Caps are the faces named `cap_*`; the wall is `wall` or `wall_*`.
2. **Open SimVascular ROM Simulation** and select the mesh in **Labelled mesh**. This can be the
   volume mesh CFD Mesh Generator made, or a labelled surface, since a 0D model needs only the
   surface. The **Boundary conditions** table lists the caps, and selecting a row highlights
   that cap in the 3D view.
3. **Give every cap a condition.** Choose its kind in the **Type** column. Its values then get
   a box each in **Values**, labelled with the value's name; hover over a label or a box for what
   the value is and its units. A value is taken when you press Enter or leave its box.
   - **Inflow**: drop the waveform's `.flow` file on Slicer, which loads it as a table, and
     choose that table in the row's **Flow table** column. The waveform is drawn under the
     table; tick **Invert** if it is upside down (see *Flow files and their sign* below). For a
     steady flow instead, type it in **Q**, positive into the model. The waveform read from the
     table is saved on the mesh with the scene, so deleting the table later does not change the
     condition.
   - **RCR**: **Rp**, **C**, **Rd** and **Pd**, i.e. proximal resistance, compliance, distal
     resistance and distal pressure. **Pd** can be left empty, and is then 0.
   - **Resistance**: **R**. A resistance drains to zero pressure and has no distal pressure, as
     svMultiPhysics' Resistance condition has none and the same conditions are written for the
     3D solver; an outlet that needs a pressure to drain to is an RCR.

   Values are in cgs units: flow in mL/s, resistance in dyn·s/cm⁵, compliance in cm⁵/dyn,
   pressure in dyn/cm² (1 mmHg = 1333.22 dyn/cm²). A value that cannot be read is shown in red in
   its box; hover over it to see why. Changing a cap's kind empties its boxes: another kind's
   values are other values.
4. **Choose the source** in the **Centerlines** section. The centerlines are traced from this
   cap, and it needs an inflow; the largest cap is suggested. On a model with several inflows
   any of them will do: the other inflows are prescribed where their own centerlines end, and
   the answer does not depend on which one is the source. **Compute centerlines** (optional)
   shows the branch-split centerlines, so you can check that every cap was reached. Running the
   simulation computes them if they are not up to date.
5. **Set up the simulation.** Choose an **Output folder**; by default it is `rom/<mesh>` next to the
   saved scene. Check the blood density and viscosity, and the number of **Cardiac cycles** and
   **Time points per cycle**. Only the last cycle is reported, so the cycles before it give the
   compliances time to settle. The mesh is taken to be in millimetres, which is what Slicer's
   coordinates are, and is scaled to the solver's centimetres on the way.
6. **Create solver files** (optional) writes the boundary condition files and svZeroDSolver's
   `solver_0d.json` into the output folder without running anything, computing the
   centerlines first if they are not current. Look at them, or change `solver_0d.json` by hand.
7. **Run simulation.** This runs svZeroDSolver on the solver files as they are, so an edit made by
   hand since creating them is what gets solved. The files are created first if they are not
   there, or if anything in the panel has changed since they were. The status line reports the
   total flow in and out over the last cycle, which should match. Select caps in the
   **Boundary conditions** table to plot them (the source is plotted when nothing is selected);
   the centerlines in the 3D view are coloured by cycle-averaged pressure. The results model also carries the flow, which the Models module can
   colour it by instead. **Export results** writes a CSV of every cap's flow and pressure over
   the last cycle: a time column, then a flow (mL/s) and a pressure (mmHg) column per cap. Flow
   is positive the way the cap's condition drives it, into the model at an inflow and out of it
   at an outlet.
8. **Show network** draws the 0D network in a dock beside the views. Every boundary condition,
   vessel segment and junction is a block, coloured by mean pressure on the same scale as the
   centerlines, with arrows from the source outwards; the caps line up in the last column. A
   vessel is labelled by its branch, and a letter for its segment if the branch has several:
   `25b` is `branch25_seg1`. Scroll to zoom and drag to move. Clicking a block shows its values
   (R, C and L for a vessel, the condition's for a cap) and its mean flow and pressure under the
   graph:
   - **A cap** selects its row in the table, which plots it and highlights it in the 3D view;
     selecting a row in the table selects its block in turn.
   - **A vessel** plots the pressure at its two ends, and picks out the stretch of centerline it
     was cut from, as a yellow sleeve around the results.

   The dock can be moved to the other side of the window, or dragged out of it onto a second
   screen.

The conditions and the source are saved on the mesh, so they come back when you reopen the
scene, and two anatomies in one scene keep separate conditions. The conditions are shared with
[SimVascular MultiPhysics](SimVascularMultiPhysics.md): set in either panel, they are the ones
both write, so a 0D model tuned here is the 3D case written there. An output folder that already
holds results shows them again when the mesh is selected.

### What gets written

```
<output folder>/
  mesh-complete/                the faces, as Mesh Prep's Export writes them
    mesh-complete.exterior.vtp
    mesh-surfaces/cap_*.vtp
  centerlines.vtp               branch-split centerlines
  inflow.flow                   the source's flow
  flow.dat, cap_*.flow          the other inflows
  rcrt.dat, resistance.dat      the outlets
  solver_0d.json                svZeroDSolver's input
  branch_results.csv            svZeroDSolver's output, per vessel segment
  cap_results.csv               Export results' suggestion: the same results, per cap
```

The boundary conditions in `solver_0d.json` are named after their faces (`RCR_cap_lpa_a`,
`FLOW_cap_azygous_vein`), except the source's, which is `INFLOW`. The solver's results are per
vessel segment, named `branch<b>_seg<s>`, which is why their file is `branch_results.csv`; the
panel works out which vessel end each cap is, and **Export results** writes the same results by
cap name, suggested as `cap_results.csv` beside it.

### Several inflows

A Fontan has the venae cavae, the azygous and the hepatic veins all entering, and the pulmonary
arteries all leaving. The centerlines run from the source, so every other inflow sits at a
centerline *end*. There it is prescribed as a flow entering the model. The resistances and
junctions of a 0D network do not care which way flow goes, so this is the same model as one
drawn from any other inflow.

Every inflow has to cover the same cardiac cycle. A steady inflow is spread over the period of
the waveforms next to it.

### Flow files and their sign

**A `.flow` file dropped on Slicer**, or opened with Add Data, loads as a table named after the
file, with a `time` column in s and a `flow` column in mL/s: the module registers a reader for
the extension, which Slicer otherwise has none for. A file is a time and a flow on each line;
lines starting with `#` are skipped, and so is the `points modes` line svMultiPhysics writes
first (`256 16`). The reader asks nothing else of the numbers, so a file that would not do as an
inflow still opens, to be looked at in Slicer's table view. The table is marked
`SimVascularROM.FlowFile` = `1`. That mark is what the **Flow table** selector offers tables by,
so a results table or one made by hand is not offered. It is not the file's path, and nothing
else saved with the scene is either: a path can carry a patient's name or ID. An inflow read
from a file by a script keeps the file's name alone.

**A table chosen for a cap is read into the cap's condition**, and asked then for what an inflow
needs: one cycle, starting at 0, its times increasing. A table that fails is shown in red, with
the reason as the tooltip. The choice is kept as a reference on the mesh, one per cap, so Slicer
keeps it through saving, reopening and importing. The inflow follows its table: edit the table
in Slicer's table view and the cap's waveform is read from it again. Choosing **None** clears
the inflow. Typing a steady flow over it, or giving the cap another kind of condition, leaves the
table behind.

The waveform is also kept on the mesh, at full precision, beside the reference. That copy is
the one used, not a reread of the table, for two reasons:

- Slicer writes a table's numbers to six significant figures when it saves a scene, and has no
  setting for more, so a table saved and reopened is not quite the file it was read from.
- A table deleted from the scene leaves the condition as it was, as a file deleted after loading
  did: a scene tidied of its tables would otherwise lose every inflow set up from them. The
  selector then shows None, and the waveform stays, described in the values.

Both panels watch the tables their mesh's inflows are read from, the hidden one too. So a table
edited is read into the conditions as the mesh has them at that moment, not as the panel last
saw them; written back whole, a hidden panel's would undo what the other panel had changed since.

**Each panel shows an inflow in its own solver's sign.** svZeroDSolver takes an inflow as
positive. svMultiPhysics measures a face's flow along its outward normal, so it takes an inflow
as negative, and its own `.flow` files have it so. Nothing in a file says which solver it was
written for.

| | SimVascular ROM Simulation | SimVascular MultiPhysics |
|---|---|---|
| An inflow is shown, drawn and typed | positive | negative |
| A file written for svMultiPhysics | Invert ticked | as it is |
| A waveform written for a 0D solver | as it is | Invert ticked |

Behind both is one condition per cap, kept on the mesh positive into the model, which is what
`svromsetup` and `svmpsetup` take. A cap given 5 mL/s in one panel is -5 mL/s in the other, and
nothing has to be turned round when a case moves between them. `svmpsetup` negates an inflow
only as it writes it for svMultiPhysics, so the file it writes is the curve the 3D panel draws.

**Invert** multiplies every flow by -1. It says whether the curve is its table's turned round for
*that panel's* solver, so the same cap can be ticked in one panel and not the other. It is
unticked as a table is chosen, and kept with the condition. Only an inflow read from a table has
the box: a steady flow is typed in the panel's sign.

When an inflow's mean goes out of the model, the text under the curve turns orange and a warning
sign shows beside its Invert box, so that a cap the wrong way round shows in its row without
being selected.

Against one sign in both panels: whichever sign it were, the files written for one of the two
solvers would show backwards, and need Invert, in the panel that solves them. Simpler still would
be svMultiPhysics taking an inflow as positive; until it does, the sign is turned round in one
place, as `svmpsetup` writes.

Drawn under the table rather than in a window that asked first, as the first version did: a
window shows a waveform once, as it is chosen, where one under the table is there for every cap
whenever its row is selected, and a box in the row says, afterwards, which way round each one
was taken.

Asked rather than guessed. The first version turned round any waveform whose mean was negative,
which suited every waveform it was tried on and would be wrong, without a word, for one that is
mostly backflow -- and "turned round" in a status line showed nobody what had been done to their
numbers.

## Setting up

Install **SlicerVMTK** from the Extensions Manager.

[svromutils](https://pypi.org/project/svromutils/), which builds the 0D model, needs no setting
up: the first time you compute centerlines or create solver files, Slicer asks to install it from
PyPI, and does. It is installed *without* the packages it says it depends on, all of which Slicer
has already (see [svromutils under Slicer](#svromutils-under-slicer)). **Do not**
`pip_install("svromutils")` into Slicer by hand: that installs them all over Slicer's own. If you
need to install it yourself, `slicer.util.pip_install("--no-deps svromutils")` in the Python
console is the same thing.

svZeroDSolver needs no setting up either. It is
[svzerod](https://pypi.org/project/svzerod/) on PyPI, and the first time you run a simulation,
Slicer asks to install it, with everything it asks for. It comes as prebuilt wheels for every
platform, so nothing is compiled, and nothing it brings replaces anything of Slicer's. By hand,
it is `slicer.util.pip_install("svzerod")`. A build of svZeroDSolver of your own, or one on the
PATH, is not used: see [Why the solver is a process](#why-the-solver-is-a-process).

Nothing else is needed: the panel has no paths to set.

## Developers

### The split

`SimVascularROM.py` is an MRML adapter over `svromsetup`, the package beside it, which imports
nothing from Slicer: the conditions and SimVascular's files for them, the case folder, the calls
into `svromutils` and the checks on what it wrote, the solver run, and the results read
back. A case set up from a terminal calls the same functions:

```python
from svromsetup import case, solver, results
from svromsetup.boundary_conditions import Inflow, Resistance

case.write_surfaces(volume_mesh, names, "case")             # {face id: name}, as Mesh Prep saves them
case.compute_centerlines("case", inlet_face_id=3)
config = case.write_solver_input("case", {"cap_RSVC": Inflow.steady(20.0), ...}, 3, "cap_RSVC",
                                 case.SimulationParameters(), "fontan")
solver.run_solver(config, "case/branch_results.csv", solver.find_solver())  # pip install svzerod
faces = results.face_results(config, solver.read_results("case/branch_results.csv"), "cap_RSVC")
results.write_face_results_csv("case/cap_results.csv", faces)
```

`svromutils` is imported only inside the two functions that call it, because it imports
VMTK. Everything else in `svromsetup` needs only numpy and VTK, and so do its tests.

### svromutils under Slicer

`svromutils` is installed with `slicer.packaging.pip_ensure`, asked to skip `vmtk`, `vtk`, `numpy`
and `scipy` (`ROM_PACKAGE_SKIPPED_DEPENDENCIES`). Each would break Slicer if pip installed it:

- **vmtk** pins a VTK of its own (`vtk==9.6.2` for 1.5.2), and a second VTK over Slicer's breaks
  the application.
- **vtk** *is* that second VTK.
- **numpy and scipy**: svromutils 0.1.2 asks for `numpy>=2.5` and `scipy>=1.18`, newer than the
  numpy 2.4 and scipy 1.17 in Slicer 5.12. pip would replace them, under every extension built
  against the old ones. What svromutils uses of them works with Slicer's own.

Skipping them installs svromutils with `--no-deps`, and removes them from its installed metadata,
so that a later `pip install` of something else does not go back and install them. The rejected
alternative was a plain `pip_ensure("svromutils")` with a constraints file pinning Slicer's own
versions. That fails outright: pip cannot satisfy `numpy>=2.5` and `numpy==2.4.*` together, and
vmtk's VTK pin conflicts with any constraint.

If the package ever declares these as optional (an extra, say `svromutils[vmtk]`), the skip list
can go.

### VMTK under Slicer

`svromutils` imports `from vmtk import vtkvmtk`, which is how pip's VMTK is laid out, and pip's
VMTK is not installed (above). SlicerVMTK carries the same
classes in modules of its own (`vtkvmtkComputationalGeometryPython`, …). So the panel assembles
a `vmtk.vtkvmtk` module out of them before importing the package (`ensureVmtk`). The branch
splitting the package relies on, `vtkvmtkPolyDataCenterlineBranchSplitting`, has been in
SlicerVMTK's VMTK since the update of September 2026.

### Why the solver is a process

svZeroDSolver is installed into Slicer's Python as `svzerod`, but not imported there. It runs as
the `svzerodsolver` command the package installs, in a process of its own. It is C++, and a
model it cannot handle is not always an exception. The released wheels compile its assertions
out, so an inconsistent model reaches Eigen unchecked. Its command line also ends in C's
`exit()` when it is called wrongly. Either one, in-process, would take Slicer and an unsaved
scene with it. A 0D model solves in well under a second, so the round trip through files costs
nothing.

The command is looked for where pip put it, in the scripts folder beside Slicer's Python
(`sysconfig.get_path("scripts")`), and never on the PATH. It runs with Slicer's own environment,
because it is Slicer's Python that runs it. Slicer's launcher points `PYTHONHOME`, `PYTHONPATH`
and the library path at Slicer's Python, which is right for that command and wrong for any other
Python's. A `svzerodsolver` found on the PATH belongs to another Python, such as a conda
environment's, so run with Slicer's environment it would start on Slicer's.

The rejected alternative was keeping the **svzerodsolver** path under Tools, for a build of
svZeroDSolver of your own. Every machine needed one, and the wheel makes it unnecessary.
A terminal workflow can still pass any executable to `run_solver`.

### The network, drawn in Slicer

**Show network** draws the 0D network in Slicer, so that nothing has to be installed to see it.
The graph and its layout are `svromsetup.network`, which reads `solver_0d.json` and needs nothing
but Python; the drawing is a Qt graphics view (`SimVascularROMLib/NetworkView.py`), which Slicer
has already. Being inside Slicer is also what makes it more than a picture: a click plots in
Slicer's own plot view and shows the block on the anatomy, which a separate application cannot do.

- **The layout is a layered tree, not Graphviz.** Graphviz is a program, not a Python package, so
  pip cannot install it, and a machine without it would draw nothing. A network traced along
  centerlines is a tree from the source, so a node's layer is its distance from the source, each
  leaf gets a row of its own, and every other node sits midway between its children. No edge
  crosses another. The caps
  at the ends are moved to one last column, because their long names, scattered through the
  layers, would widen every one of them; in one column they read like the conditions table.
- **A vessel is shown where its results are drawn.** The stretch of centerline it was cut from is
  found by the rule that colours the centerlines (`results.vessel_lines`), so the sleeve marks
  the stretch coloured by that vessel's results.
- **Clicks are found by the view**, by what is under the mouse, rather than through Qt's item
  selection, which draws a dashed box of its own over a selected item. A selected block gets a
  halo in the yellow Mesh Prep highlights a cap with, which shows on every pressure colour, where
  a yellow outline would not.

The rejected alternative was **Slicer's web view with a JavaScript graph library** (Cytoscape.js,
say). It leaves third-party JavaScript to keep in the repository, a bridge from JavaScript to
Python for every click, and a view that misbehaves on some machines, over remote desktop or
without a GPU.

### Outlets paired by position

The package matches outlet names to boundary conditions by list position. Nothing ties the
order of the centerline ends to the order of the caps, and an off-by-one list puts every
condition on its neighbour's vessel, with all the ids still valid. So the centerline ends are
paired with the cap files by where they are: one to one, closest first, and refused if an end
is further than a cap's width from its cap. `check_solver_input` then holds the written file to
what was asked: one condition per cap, on one vessel end, of the right kind, over the requested
cycles.

### Tests

The package's tests are headless (pytest, from `SimVascularROM/`):

```sh
python -m pip install svromutils svzerod  # outside Slicer, where pip's VMTK and VTK are what you want
python -m pytest
```

The solver is found beside that Python, whether or not its environment is activated. To test a
build of svZeroDSolver of your own instead, set `SVZERODSOLVER=/path/to/svzerodsolver`.

The whole-case tests, and the module's own tests under Slicer (Reload and Test), run on
`case01` of svReducedOrderModelingUtils'
[TESTING_DATA](https://github.com/SimVascular/svReducedOrderModelingUtils/releases/tag/TESTING_DATA)
release, an aorta with its tutorial's boundary conditions. `svromsetup.testing_data` downloads it
once, about 17 MB, into pytest's or Slicer's cache and checks it against the release's SHA-256
digests; offline, set `SVROM_TESTING_DATA` to a folder holding its files, and without either the
tests that need it are skipped. The tests that need `svromutils` (and VMTK) or the solver skip
themselves where these are missing; with only numpy, VTK and pytest, 43 run.

A `--testing` run loads none of the installed extensions, so pass SlicerVMTK's module folders as
`--additional-module-paths`, along with `SimVascularMeshPrep` and `SimVascularROM`.
