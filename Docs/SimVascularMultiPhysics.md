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
   10 s of 1 ms steps unless you change it (see *Defaults* below).
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
- **Time stepping and saving:** 10000 steps of 0.001 s (10 s), saving results and restart files
  every 100 steps from step 1. For a whole number of cardiac cycles, set the steps to cycles times
  steps per cycle.
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
