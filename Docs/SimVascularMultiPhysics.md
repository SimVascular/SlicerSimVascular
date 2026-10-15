# SimVascular MultiPhysics

## Summary

Writes the case [svMultiPhysics](https://github.com/SimVascular/svMultiPhysics) runs a rigid-wall
CFD simulation from: `solver.xml`, the mesh-complete folder it reads, and a waveform file per
inflow. You give it a volume mesh whose faces are named in
[SimVascular Mesh Prep](SimVascularMeshPrep.md), and a boundary condition per cap. Every other
setting has a default and can be changed.

The boundary conditions are the ones [SimVascular ROM Simulation](SimVascularROM.md) sets for the
same mesh: there is one set per mesh, edited in either panel. A 0D model tuned there is the 3D
case written here, with nothing to copy across.

The module writes the case, and runs a few steps of it here to test it; the run itself is yours,
from the output folder, on your machine or a cluster:

```sh
mpiexec -np 8 svmultiphysics solver.xml
```

A test of a few steps finds what would stop that run -- a library missing, an element the mesher
left inverted, an outlet drawing flow in, a run gone NaN -- in minutes here, rather than after a
queue on a cluster, and times the steps, which says what the full run will cost.

## Tutorial

1. **Name the faces** of the volume mesh in SimVascular Mesh Prep. A mesh out of Clip Vessel
   arrives named after its clip points, and opening it there is enough to save those names on it.
2. **Open SimVascular MultiPhysics** and select the mesh in **Volume mesh**. It has to be the
   volume mesh CFD Mesh Generator made, not the surface: a 3D simulation is solved on the volume
   elements.
3. **Give every cap a condition** in the **Boundary conditions** table, exactly as in the ROM
   panel. Any you already set there are shown here.
   - **Inflow**: double-click the values to load a waveform from a `.flow` file, or select the
     cell and type a steady flow in mL/s.
   - **RCR**: `Rp, C, Rd, Pd`; `Pd` can be left out, and is then 0.
   - **Resistance**: `R`.

   Every wall face is given a no-slip condition; there is nothing to set for them.
4. **Check the settings** under **CFD Simulation**. Each group is collapsed and starts at the
   defaults; a value you change is shown in bold, and one that cannot be read in red, with the
   reason as its tooltip. **Reset settings to defaults** puts them all back. The time stepping is
   1 ms steps unless you change it (see *Defaults* below).

   Two sections of **General** each choose how a value is set; the value not typed is shown,
   greyed:
   - **Number of time steps**: **User-defined** (10,000, the default), typed; or **Calculate from
     cardiac cycles** (10), worked out from the inflow waveform's cycle and the time step --
     changing the number of cycles chooses it. A case whose inflows are all steady has no cycle to
     count, and runs the number typed, whichever is chosen; the panel says so under the cycles.
   - **Time step size**: **User-defined** (0.001 s, the default), typed; or **Estimate time step
     from CFL**, which opens three fields under it:
     - **CFL number**, 1 recommended;
     - **Max velocity**, in cm/s, filled in with the fastest velocity an inflow imposes -- its peak
       flow, in mL/s, over its cap's area, in mm², twice that on the centreline with the Parabolic
       profile;
     - **Mesh size**, in mm, filled in with the mesh's mean edge length.

     The step is the CFL number times the mesh size over the max velocity, with the millimetres
     scaled to the solver's centimetres by **Mesh scale factor**, rounded down to a whole number of
     steps per cardiac cycle. It is worked out again whenever any of these, the mesh or the
     conditions change. Type over the max velocity or the mesh size -- for a narrowing faster than
     the inflow, a region finer than the mean -- and it is bold, used as typed and kept as the case
     changes; **Use the case's values** puts both back.

     With **Adapt the saving to the time step**, ticked by default, the saving is kept as it was in
     time -- results as many per cycle, at the same points of every cycle, saving started when it
     was, restart files as often. Type any of the three saving settings under it instead and the
     saving is yours: the box is cleared, what you typed is kept in steps as the step changes, and
     the step is only rounded to whole steps per cycle. Tick it again to have the saving adapted
     from what it is. The line under them gives every number with its units, the case's own beside
     any typed over, how many results the run saves, and -- for a saving typed -- whether results
     fall at the same point of every cycle.

5. **Choose an output folder**, by default `svmultiphysics/<mesh>` next to the saved scene, and
   click **Create solver files**. The status line says what was written.
6. **Test it here** (optional), under **Test run**:
   - **svMultiPhysics** is found where the installers put it, `/usr/local/sv/svMultiPhysics/<date>/bin`,
     on Ubuntu and macOS; at `SVMULTIPHYSICS`; or on the PATH. For a build of your own, give
     its path, such as `build/svMultiPhysics-build/bin/svmultiphysics`. On Windows, which has no
     build, the Ubuntu one is found inside WSL2 and run through `wsl.exe`. The line under the
     field says which was found. It needs OpenMPI, LAPACK and BLAS beside it, as the installers'
     page says; without `mpiexec` the test runs on one processor.
   - Set **Time steps** (5) and **Number of Processor(s)** (8), and click **Run test**. The solver files are
     written first if anything in the panel has changed since they were; otherwise they are run
     as they are, hand edits included. The panel shows the step it is on as it goes, and the
     solver's own output under **Solver output**.
   - At the end it says how many steps ran, the time per step, what the full run takes at that
     rate and how much its results come to, and what went wrong, in words: the solver's error,
     a library missing, more processors than cores, a NaN. Steps that used up their nonlinear
     iterations without converging are noted.
   - **Stop test** stops it after the step it is on; pressed again, it kills it.
   - **Show results** draws the last step: arrows along the velocity, coloured by speed, inside
     the faded mesh, and which way the flow went through each face against its condition, with
     the inflow beside its waveform's value at that time.

### What gets written

```
<output folder>/
  solver.xml                  run svMultiPhysics on this, from this folder
  mesh/                       the mesh-complete folder, as Mesh Prep's Export writes it
    mesh-complete.mesh.vtu
    mesh-complete.exterior.vtp
    walls_combined.vtp
    mesh-surfaces/<name>.vtp  one file per face
  <cap>.flow                  one per inflow waveform
```

Every path in `solver.xml` is relative to its own folder, which is the directory svMultiPhysics is
run from, so the folder can be moved, to a cluster for instance, as a whole.

A test is written beside it, never in it:

```
<output folder>-test/
  solver.xml                  the case's, cut to the test's steps; its paths point back at the case
  8-procs/                    what svMultiPhysics wrote, named after the processors it ran on
    result_005.vtu            the last step, the one Show results draws
    B_NS_Velocity_flux.txt    the flow through each face, every step
    B_NS_Pressure_average.txt the pressure on each face, every step
```

### How the conditions are written

| In the table | In `solver.xml` |
| --- | --- |
| Inflow, waveform | `Dir`, `Unsteady`, `Temporal_values_file_path`, with the inflow profile and imposed flux from the settings |
| Inflow, steady | `Dir`, `Steady`, `Value` |
| RCR | `Neu`, `RCR`, with `RCR_values` and the initial pressure from the settings |
| Resistance | `Neu`, `Resistance`, `Value` |
| (every wall face) | `Dir`, `Steady`, `Value` 0: no slip |

**Inflow is written negative.** The table takes an inflow as positive, into the model, as the 0D
solvers do. svMultiPhysics measures a face's flow along its outward normal, so flow into the
domain is negative, both in a `Value` and in a `.flow` file.

**A `.flow` file's header** is the number of time points and the number of Fourier coefficients
the solver fits the waveform with (the *Inflow Fourier coefficients* setting, 16 by default). It
is not the number of columns: two would smooth a waveform into very nearly a sinusoid.

## Defaults

All the defaults are in one table, `svmpsetup/settings.py`. The panel draws its fields from it,
and nothing else holds a default:

- **Linear solver:** for a CFD simulation the LS type is `NS` and the `Linear_algebra` type
  `fsils` (with the fsils preconditioner). These are fixed rather than settings, and the group says
  so. Its settings are the lab's rigid-wall defaults: `Max_iterations`
  10, `NS_GM_max_iterations` 200, `NS_CG_max_iterations` 500, `Tolerance` 0.4, `NS_GM_tolerance`
  0.01, `NS_CG_tolerance` 0.2, `Krylov_space_dimension` 50. There is no `Absolute_tolerance`, so
  svMultiPhysics' own 1e-10 applies. The Fontan template tightens it to 1e-17; that was considered
  and not taken.
- **Time stepping and saving:** a user-defined 0.001 s step and a user-defined 10,000 steps
  (10 s), saving results and restart files every 100 steps from step 1. Calculated from 10 cardiac
  cycles instead, the steps are 9,091 for the tutorial aorta's 0.909 s cycle. On the aorta, 1 ms is
  CFL 4.7; estimated from CFL 1 the step is 2.1e-4 s, saving every 477, and the 10,000 steps then
  run 2.1 s -- 42,930 for 10 cycles.
- **Fluid iterations:** `Min_iterations` 3, `Max_iterations` 10 and `Tolerance` 1e-4, the lab's,
  over the Fontan template's 2, 5 and 1e-3: at those, the test run of the tutorial aorta used all
  five iterations in three of its first five steps without reaching the tolerance. At these, all
  five steps converged, at 23 s a step on 8 processes.
- **Everything else**, for now: the Fontan workflow's `solver_template.xml` (in ClinicalWorkflows),
  without its scalar-transport equation, as a stand-in until the defaults for this panel are
  settled.
- **Mesh scale factor:** 0.1. Slicer's coordinates are millimetres and the solver works in
  centimetres. It is always written: left out, the solver takes 1.0, and a mesh in mm becomes a
  domain ten times too large.
- **RCR initial pressure:** 0, which starts the outlets empty.

## Developers

### The split

`SimVascularMultiPhysics.py` is an MRML adapter over `svmpsetup`, the package beside it, which
imports nothing from Slicer:

- `settings` is the defaults table, and how a typed value is read;
- `solver_xml` builds the file element by element;
- `case` writes the folder and checks it.

A case written from a terminal calls the same functions:

```python
from svmpsetup import case
from svromsetup.boundary_conditions import Inflow, RCR

case.write_case("run", volume_mesh, names,          # {face id: name}, as Mesh Prep saves them
                {"cap_inlet": Inflow.steady(10.0), "cap_outlet": RCR(121, 1.5e-4, 1212)},
                {"general.number_of_time_steps": 4000})
```

The mesh goes through `svmeshcomplete` and the conditions are `svromsetup`'s, so a case written
here, one packaged by Mesh Prep, and one set up in the ROM panel agree on every face name and
every condition.

### Built, not templated

The file is built from the settings and the conditions rather than by filling in a template. What
a template carries besides its blanks is the most case-specific content in a `solver.xml`: its
equations, numerics and outputs. Applied to the wrong case, a template converges on the wrong
physics with nothing said, which is why the Fontan workflow's writer refuses to ship a default
one. Here each element comes from one setting or one condition, and each setting is one row of
one table.

### Why every value is parsed

svMultiPhysics reads a number with `istringstream`, which takes the longest numeric prefix and
drops the rest without a word: `1,5e-4` is read as `1`. So nothing typed is written as typed. A
value is parsed into a number of its setting's kind, refused in red if it is not one, and written
from the number.

### What the writer checks

The solver refuses some of these only once a job is running on a cluster, and others not at all.
So `case.check_case` reads the written file back and refuses it if:
- a face of the mesh is not listed (a wall face left out is a hole);
- a listed face has no condition;
- a face file or a waveform is missing.

### One table of conditions

The table is `SimVascularROMLib.BoundaryConditionsTable`, beside the ROM module, which both panels
use. Two copies would be two tables that drift. The conditions are kept on the mesh node, under
the attribute the ROM panel first kept them in, so scenes saved before this panel existed still
open with theirs.

### The test run

`svmpsetup.trial` is the test and how the solver's output is read; the panel runs the command it
builds as a process of its own and feeds it what the solver prints.

- **A copy of the written file, not the case written again.** The test is the case's own
  `solver.xml` with the step count and the saving changed and nothing else, its paths pointed back
  at the case's mesh and waveforms. So what passes is the file that goes to the cluster, hand edits
  and all. Writing the test from the panel instead would test a file that is not the one run.
- **Beside the case, not in it.** In the case folder, a test's 40 MB results would go to the
  cluster with the case.
- **Only its last step saved.** The full run saves one step in a hundred; a test saving every one
  would be timed with a 40 MB result written in every step, which the full run does not do, and
  its estimate of the full run would be too long by that much. The last step is the one shown. The
  faces' flow and pressure are written every step regardless.
- **With the environment Slicer was started with**, `slicer.util.startupEnvironment()`, not the one
  its launcher made: that one's library folders, Slicer's VTK among them, are no business of a
  program linked against its own.

svMultiPhysics' report says less than it looks as though it does, which is why it is read here:

- The `s` after `<step>-<iteration>` marks a step whose results were *saved*, not one that
  converged.
- A step that uses up `Max_iterations` is accepted as converged without a word. The test notes
  such a step: its last iteration at the limit, with its residual, against both the step's first
  iteration and the run's, still above `Tolerance`.
- Nothing checks for NaN except an RCR outlet's integration, so a run gone NaN carries on writing
  NaN. The test stops it at the first NaN and says so.
- The solver's error goes to stderr from the first process only. Raised on another, it shows only
  as MPI's abort, which the test says, without a reason it does not have.

The direction of the flow is the solver's own: `B_NS_Velocity_flux.txt` is the velocity
integrated over each face along its outward normal, so an inflow is negative. Results are written
in the mesh's own coordinates, millimetres, so the arrows sit in the anatomy as they are; the
velocity is in the solver's units, cm/s.

Running through WSL is the one path not tested here, for want of Windows: the solver is found by
asking WSL's shell where it is, and run with `wsl.exe --cd <folder> --exec`, which translates a
Windows folder itself.

### The number of time steps and the time step: each chosen, or worked out

Each is a section of its own with a choice of how it is set -- the steps calculated from cardiac
cycles or user-defined, the step user-defined or estimated from the CFL -- one or the other, never
both: two fields that each set one value would disagree the moment either was changed. What is
worked out is shown, greyed and read-only, and not bolded, since nobody typed it; it is what
`solver.xml` gets. The cycles and the CFL number are not settings, since nothing in the file is
them: they are kept on the scene, with which way each value is set.

- **The steps user-defined by default**, 10,000, as the case was always written: the user's
  choice, after a first version that calculated them from 10 cycles by default and greyed the
  field a user expected to type in. The cycles are there because a run is read by the cycle -- the
  last one, or the last few, once the RCR outlets have filled -- and a number of steps is a number
  of cycles only by arithmetic that has to be redone whenever the time step changes. User-defined,
  the steps are the user's, and the CFL estimate leaves them alone, whatever length of time they
  then run; it says what that is. A steady inflow has no cycle to count, so with one the steps are typed whichever is
  chosen: greyed, they were a leftover from the last waveform that nobody could change.
- **The step user-defined by default**, the value the case was always written with. Estimated, it
  is worked out again as the CFL, the mesh, the conditions or the profile change, rather than at a
  button press, so that what is written is never an estimate for a mesh or an inflow since
  replaced. It is worked out first, and the steps from it: the steps follow the step.
- **What the estimate is made from is shown, and can be typed over.** The max velocity and the mesh
  size are filled in from the case when the estimate is chosen, because those are the two numbers a
  user has reason to doubt: the inflow is not always where the flow is fastest, and the mean edge is
  not the finest. A value typed over is the user's: bold, used as typed rather than as the case's,
  saved with the scene, and kept when the mesh or the conditions change, which a value left to the
  case follows. Choosing the estimate again, or **Use the case's values**, puts both back. The
  estimate uses a value left to the case as worked out, not as its field rounds it.
- **User-defined first in both sections**, and the default in both, so that they read alike.
- **Nothing a user might reach for is greyed without a word.** The cycles box takes a number whether
  or not its option is chosen, and changing it chooses it; with no cycle to count -- a steady
  inflow, no mesh, a time step that does not read -- the cycles say why under them. **Use the
  case's values** can always be clicked, and fills both in from the case again. Both were disabled
  until they would do something, and read as broken: a greyed box gives no reason, and the user
  could not tell that it wanted a radio button pressed first.

A scene opens with its steps calculated from cycles only if it was saved so; one saved before the
choice existed opens with them user-defined, as they were written, so that opening it does not
quietly make its run another length. Its time step opens user-defined.

### The time step from the CFL

`svmpsetup.time_step` is the estimate; the panel measures the mesh and the caps, and sets the
fields. `dt = CFL * h / u`, in the solver's units, decided as follows:

- **u is the fastest velocity an inflow imposes**: its waveform's peak flow over its cap's area,
  doubled with the Parabolic profile, because a Poiseuille profile peaks on its centreline at twice
  its mean. Not only on a circular cap: svMultiPhysics' profile is `1 - r²/R(θ)²`, with `R(θ)` the
  distance from the cap's centroid to its rim in each direction, scaled to the flux, and that
  averages half its peak over any cap whose rim the centroid sees all of. Checked against the
  solver with a steady 100 mL/s through the aorta's 270.9 mm² valve: 73.84 cm/s estimated, 73.91
  imposed in svMultiPhysics' first step, the difference the discrete cap's. A steady inflow is its
  one flow throughout, and is said to be steady rather than a peak. The mean, peak flow over area, was rejected: with the default profile it gives a step
  twice too long. With **Impose flux** off an inflow's values are velocities, not flows, and the
  estimate is refused rather than made as though they were flows. Over several inflows, the
  fastest.
- **h is the mean edge length** of the volume mesh, each edge counted once however many cells
  share it, so a labelled mesh's face triangles do not count theirs twice. It is about the size the
  mesh was made at: CFD Mesh Generator's 1 mm target gives 0.73 mm on the aorta. Rejected: the
  smallest edge, set by the worst few of a million elements (0.28 mm there), which would ask for a
  step two and a half times shorter than the rest of the mesh needs; and the edges at the inflow
  cap, which ignore faster flow downstream, through a narrowing for one.
- **Either can be set instead.** These are what **Max velocity** and **Mesh size** are filled in
  with, and `estimate` takes the same two: `velocity`, the case's when left `None`, and
  `edge_length`, with `mesh_edge_length` the mesh's own, given only so that the explanation can say
  it beside one typed over. A velocity set by hand needs no inflow, so a case with Impose flux off
  can be estimated after all, from a velocity someone knows.
- **The units go through the case's own scale factor.** A flow in mL/s is already in the solver's
  cm³/s, but the cap's area is in mm² and the edges in mm, as Slicer's coordinates are, so they are
  scaled by **Mesh scale factor** -- squared for the area -- not by a fixed 0.1: a case written with
  a changed factor is estimated as it will run.
- **Rounded down to whole steps per cycle**, by the period the conditions are written with (the
  first waveform's), and the run that many steps a cycle times its cycles, when it is calculated
  from them: results then fall at the same point of every cycle, and a run ends on a cycle's end. Down, never up, so the CFL is at most the one asked for. A steady inflow
  has no cycle, and keeps its simulated time to the nearest step instead.
- **The saving is kept in time, not in steps.** Every saving setting counts steps, so a step five
  times shorter at the same "every 100 steps" would save five times as many results -- 472 rather
  than 100 on the aorta, 40 MB each -- and "start saving after step 5000" would start at a fifth of
  the time. So results are saved as many times per cycle as they were, and the steps per cycle are
  rounded up to a multiple of that, so each result lands at the same point of every cycle: an
  interval that does not divide the cycle drifts through it, and no two cycles compare result for
  result. On the aorta, 4,287 steps a cycle become 4,293, nine results a cycle of 477 steps each,
  and the CFL 0.9986. Saving fewer than once a cycle stays every so many whole cycles. The start of
  saving is kept at the same time, and restart files as often in time, neither tied to the cycle.
  Leaving the saving as it was, in steps, was the first version, and was rejected for the results
  it multiplied. Worked out again with nothing changed, the estimate keeps the saving it set, and
  says nothing was replaced.
- **Or the saving is as typed.** Adapted, a saving typed was taken as the saving before and adapted
  again the next time anything changed: on the aorta, every 1000 steps became every 1191 at CFL 0.9
  and 1072 back at 1 -- and the same on opening the scene again. So typing a saving setting while
  the estimate adapts it clears **Adapt the saving to the time step**, kept on the scene, and the
  saving is left in steps exactly as typed; `estimate(..., adapt_saving=False)`. With no saving to
  fit, the step is rounded to whole steps per cycle alone, and results that then fall at a
  different point of each cycle are said rather than corrected, since someone typed them. A box
  rather than each saving field bolded when typed, as the max velocity and the mesh size are: the
  three are one choice -- adapted, or the user's -- and the box shows which, and gives the adapting
  back. Only a field changed clears it, since one left unchanged still finishes its editing.

### Its own package first

Installed, this module and `svmpsetup` are in one folder. Loaded from a checkout, through the module
paths, with the extension installed as well, the installed `svmpsetup` comes first on the Python
path, because Slicer's launcher puts the extension's folder there before any module is found. So
the module puts its own folder first before importing it: otherwise a checkout's module runs the
installed package, and fails on anything the checkout added. The ROM panel and Mesh Prep do not yet
do the same for `svromsetup` and `svmeshcomplete`.

### Not here

- **FSI.** The panel writes rigid-wall CFD only, which is what the class runs.
- **The full run.** The module tests a few steps of the case; the run itself is done elsewhere.

### Tests

The package's tests are headless (pytest, from `SimVascularMultiPhysics/`). Two of them run
svMultiPhysics on `svmeshcomplete`'s cube when they find it at `SVMULTIPHYSICS` or on the PATH:
the case for two time steps, and a test of it, read as the panel reads it. The test's reading of
the output is held to lines svMultiPhysics printed for the tutorial's aorta, and to the error,
MPI and loader messages it recognises:

```sh
SVMULTIPHYSICS=/path/to/svmultiphysics python -m pytest
```

The module's tests (`SimVascularMultiPhysicsTest`) run under Slicer, with this module, ROM and
Mesh Prep on `--additional-module-paths`. One of them runs a test of the cube through the panel,
and shows its results, when the solver is found.
