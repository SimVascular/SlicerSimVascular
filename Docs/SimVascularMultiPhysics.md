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

The module writes the case and does not run it. Run it from the output folder, on your machine or
a cluster:

```sh
mpiexec -np 8 svmultiphysics solver.xml
```

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

### Not here

- **FSI.** The panel writes rigid-wall CFD only, which is what the class runs.
- **Running the case.** The module writes it; running is done elsewhere.

### Tests

The package's tests are headless (pytest, from `SimVascularMultiPhysics/`). One of them runs
svMultiPhysics for two time steps on `svmeshcomplete`'s cube when it finds the solver at
`SVMULTIPHYSICS` or on the PATH:

```sh
SVMULTIPHYSICS=/path/to/svmultiphysics python -m pytest
```

The module's tests (`SimVascularMultiPhysicsTest`) run under Slicer, with this module, ROM and
Mesh Prep on `--additional-module-paths`.
