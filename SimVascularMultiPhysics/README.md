# SimVascular MultiPhysics

Write the case svMultiPhysics runs a rigid-wall CFD simulation from: `solver.xml`, the
mesh-complete folder it reads, and a waveform file per inflow. You give it a volume mesh whose
faces SimVascular Mesh Prep named, and a condition per cap (the same conditions SimVascular ROM
Simulation sets for that mesh). Everything else has a default and can be changed.

```
<output folder>/
  solver.xml           mpiexec -np 8 svmultiphysics solver.xml, from here
  mesh/                the mesh-complete folder, as Mesh Prep's Export writes it
  <cap>.flow           each inflow waveform: header, then time and flow, inflow negative
```

**Number of time steps** is user-defined by default, or calculated from cardiac cycles; **Time step
size** is user-defined, or estimated from a CFL number (1 recommended), a max velocity and a mesh size
-- filled in from the fastest inflow and the mesh's mean edge length, and open to typing over --
rounded to whole steps per cardiac cycle, with the saving kept as often in time as it was, or left as
you type it.

**Test run** runs a few steps of the case on this machine first, in `<output folder>-test` beside it,
with the svMultiPhysics the installers put in `/usr/local/sv/svMultiPhysics`, a build of your own,
or, on Windows, the one in WSL2. It reports the time per step and what the full run will cost, and
any error in words, and can draw the last step's velocity in the anatomy.

## Outside Slicer

The writing is [`svmpsetup/`](svmpsetup). It imports numpy, VTK, `svmeshcomplete` and
`svromsetup`'s boundary conditions, and nothing from Slicer:

```bash
python -m pip install -e ../SimVascularMeshPrep -e ../SimVascularROM -e .
SVMULTIPHYSICS=/path/to/svmultiphysics python -m pytest
```

## Full documentation

[Docs/SimVascularMultiPhysics.md](../Docs/SimVascularMultiPhysics.md) covers the panel step by
step, how each condition is written, where every default comes from, why the file is built
rather than templated, and how a test run reads what svMultiPhysics prints.
