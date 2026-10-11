# SimVascular ROM Simulation

Set up a 0D simulation of a vascular model whose faces SimVascular Mesh Prep named, run it with
svZeroDSolver, and look at what it says: per cap, over the cycle, and along the centerlines
inside the anatomy.

## Why this is a step at all

A 0D model is cheap to solve and expensive to set up. The centerlines have to be traced from an
inlet and split into branches, every cap needs a boundary condition under the name its face
goes by, and the solver's results are per vessel segment, `branch14_seg2`. Someone then has to
work out which of those is the right pulmonary artery. This panel does the bookkeeping on both
sides: the conditions in, by face name; the results out, by face name and in place; and the
network it solved, drawn as a graph whose blocks can be clicked for their values and results.

```
case/
  mesh-complete/              the faces, as Mesh Prep's Export writes them
  centerlines.vtp             branch-split, from the inlet
  inflow.flow, flow.dat       the inflows
  rcrt.dat, resistance.dat    the outlets
  solver_0d.json              svZeroDSolver's input
  branch_results.csv          its output, per vessel segment
  cap_results.csv             the same per cap, from Export results
```

## Outside Slicer

The setting up is [`svromsetup/`](svromsetup), which imports numpy, VTK and `svmeshcomplete`,
and imports [`svromutils`](https://pypi.org/project/svromutils/) only where it is called. The
`rom` extra brings it, and svZeroDSolver as [`svzerod`](https://pypi.org/project/svzerod/):

```bash
python -m pip install -e ../SimVascularMeshPrep -e ".[rom]"
python -m pytest
```

Under Slicer, the panel installs both itself, the first time each is needed, so nothing is
installed by hand. `svzerod` is installed whole. `svromutils` is installed without the VMTK, VTK,
numpy and scipy it asks for: Slicer has its own of each, and pip's would replace them.

## Full documentation

[Docs/SimVascularROM.md](../Docs/SimVascularROM.md) covers the panel step by step, several
inflows, the sign of a flow file, what it installs and why VMTK and the solver are reached the
way they are, and how the network is drawn.
