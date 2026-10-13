"""Set up a 0D simulation of a labelled vascular model, run it, and look at what it says.

A reduced-order model replaces the three-dimensional flow with a network: the vessels are cut
along their centerlines into segments, each segment is a resistance, an inductance and a
capacitance worked out from its length and cross-section, and the network is closed by one
boundary condition per cap. svZeroDSolver solves it in well under a second, which is what makes
it the model to tune boundary conditions on, check a 3D case against, or run where a 3D one
cannot be afforded.

This panel is the setting up and the looking. The mesh is one whose faces SimVascular Mesh Prep
has named -- the names are read off it, never typed again here -- and the conditions are set per
cap in a table, an inflow's waveform loaded by double-clicking its values. The centerlines come
from SlicerVMTK, traced from the inlet the operator picks; the model is built and written by
`svromutils`; svZeroDSolver (`svzerod`) is run as a process; and what it says comes back as a plot
over the cycle of the caps selected in the table, the centerlines coloured by pressure, laid
inside the anatomy they came from, a CSV of every cap to take away, and the 0D network itself,
drawn as a graph whose every block can be clicked for its values and results.

## Where the work is

In `svromsetup`, the package beside this file, which imports nothing from Slicer: the conditions
and SimVascular's files for them, the case folder, the calls into `svromutils` and the
checks on what it wrote, the solver run, the results read back. This file is the MRML adapter --
it reads the selected node, calls the package, and puts the answer in the scene -- so a case set
up here and one set up from a terminal are the same case.

## Several inflows

The centerlines are traced from one inlet, so a model with several inflows -- a Fontan, with the
venae cavae and the hepatic veins all entering -- has the others at centerline *ends*. They are
prescribed there as flows entering the model, which the 0D network does not mind: a vessel's
resistance and a junction's mass balance do not care which way the flow goes. Which inflow is
the inlet makes no difference to the answer, only to how the network is drawn.

## VMTK and the packages under Slicer

Nothing this panel runs is installed by hand or kept anywhere of its own: both packages are on
PyPI, and each is installed into Slicer's Python the first time it is needed, through Slicer's
own installer, which asks first.

`svzerod`, svZeroDSolver, is installed whole. It is a compiled extension, but published as wheels
for every platform and every Python Slicer has shipped, so nothing is built; and what it asks for
-- numpy, unpinned, and pandas -- replaces nothing of Slicer's. Its `svzerodsolver` command is run
as a process, from where pip put it beside Slicer's Python and with Slicer's own environment,
which that Python needs; `svromsetup.solver` says why it is not imported.

`svromutils`, the package that builds the model, is installed -- but without what it declares it depends on, every one of which is
in Slicer already and none of which may be installed over it. pip's VMTK pins a VTK of its own,
and a second VTK over Slicer's breaks the application; `vtk` is that second VTK; and its numpy and
scipy floors are newer than the numpy and scipy Slicer and its extensions were built against.

It imports VMTK as `from vmtk import vtkvmtk`, which is how pip's VMTK is laid out. SlicerVMTK
carries the same classes in modules of its own, so `vmtk.vtkvmtk` is assembled from those before
the package is imported -- see `ensureVmtk`.
"""

import importlib
import json
import logging
import os
import sys
import time
import types

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from svmeshcomplete.face_table import sanitized
from svromsetup import boundary_conditions as bcs
from svromsetup import case, network, results, solver

# The faces and the table of their conditions, shared with SimVascular MultiPhysics -- the names
# are Mesh Prep's and the conditions one set per mesh, whichever solver they are written for.
from SimVascularROMLib.BoundaryConditionsTable import (  # noqa: F401 (the tests use these)
    BOUNDARY_CONDITIONS_ATTRIBUTE,
    FACE_NAMES_ATTRIBUTE,
    INVALID_VALUES_COLOR,
    KIND_CHOICES,
    TYPE_COLUMN,
    VALUES_COLUMN,
    BoundaryConditionsTable,
    NamedMeshFaces,
    clearHighlight,
)
from SimVascularROMLib.NetworkView import NetworkView

# The cap the centerlines start at, kept on the mesh beside its conditions. The 0D panel's own:
# a 3D model has no source.
INLET_FACE_ID_ATTRIBUTE = "SimVascularROM.InletFaceID"

# The nodes a run leaves in the scene, referenced from the mesh they belong to so that each
# anatomy keeps its own, and so that they are found again in a saved scene.
CENTERLINES_REFERENCE = "ROMCenterlines"
RESULTS_MODEL_REFERENCE = "ROMResultsModel"
RESULTS_TABLE_REFERENCE = "ROMResultsTable"
RESULTS_CHART_REFERENCE = "ROMResultsChart"
# One vessel's results, filled again whenever a vessel is clicked in the network, so that plotting
# a vessel is plotting a table like a cap is, and clicking through fifty of them leaves one table
# and one pair of plot series rather than fifty.
VESSEL_TABLE_REFERENCE = "ROMVesselTable"
VESSEL_COLUMNS = (("inlet", "pressure_in"), ("outlet", "pressure_out"))

# The stretch of centerline a vessel clicked in the network was cut from, drawn as a sleeve round
# the results' tubes. Not saved with the scene: it is a way of looking, not something the case has.
SELECTED_VESSEL_NODE_NAME = "ROM selected vessel"
SELECTED_VESSEL_RADIUS_FACTOR = 1.25
SELECTED_VESSEL_OPACITY = 0.6
NETWORK_DOCK_OBJECT_NAME = "SimVascularROMNetworkDock"

# The rest of the panel's state, which is the scene's.
INPUT_MESH_REFERENCE = "InputMesh"
CASE_DIRECTORY_PARAMETER = "CaseDirectory"
DENSITY_PARAMETER = "Density"
VISCOSITY_PARAMETER = "Viscosity"
CYCLES_PARAMETER = "CardiacCycles"
POINTS_PARAMETER = "PointsPerCycle"


# The units the mesh is in, which are Slicer's: its coordinates are millimetres whatever unit it
# displays them in. Both solvers work in centimetres, and the package scales lengths by 0.1 and
# areas by 0.01 on the way.
MESH_UNITS = "mm"

TIME_COLUMN = "time [s]"
PLOT_COLUMN_ATTRIBUTE = "SimVascularROM.Column"
# What the centerlines are coloured by and the caps are plotted in. Pressure, because it is what a
# 0D model is run to find -- the flows at the caps are mostly what the conditions set -- and the
# one quantity that varies along every branch. Flow is in the export, and on the results model
# as an array of its own for the Models module to colour by.
RESULTS_QUANTITY = "Pressure"
RESULTS_ARRAY_NAME = results.PRESSURE_ARRAY_NAME

# svromutils, the package that builds the 0D model, as pip is asked for it -- and what it declares
# it depends on that is not installed with it. Every one of those is in Slicer already, and
# installing any of them would break it: vmtk brings a VTK of its own and vtk is one, a second
# VTK over Slicer's; and svromutils asks for numpy>=2.5 and scipy>=1.18, newer than the ones
# Slicer and its extensions were built against, which pip would replace them with. What it uses
# of them works with Slicer's own, VMTK through SlicerVMTK (see ensureVmtk).
ROM_PACKAGE_REQUIREMENT = "svromutils"
ROM_PACKAGE_SKIPPED_DEPENDENCIES = ["vmtk", "vtk", "numpy", "scipy"]

# The modules SlicerVMTK's classes are in. pip's VMTK gathers the same classes into one module,
# `vmtk.vtkvmtk`, and that is what `svromutils` imports.
VMTK_MODULES = (
    "vtkvmtkCommonPython",
    "vtkvmtkComputationalGeometryPython",
    "vtkvmtkDifferentialGeometryPython",
    "vtkvmtkIOPython",
    "vtkvmtkMiscPython",
    "vtkvmtkSegmentationPython",
    "vtkvmtkITKPython",
    "vtkvmtkContribPython",
)

# What a results tube is drawn in: blue low to red high, which is how pressure is read.
RESULTS_COLOR_NODES = ("vtkMRMLColorTableNodeFileColdToHotRainbow.txt",
                       "vtkMRMLColorTableNodeRainbow")

# What the mesh is faded to when centerlines or results are shown inside it.
RESULTS_MESH_OPACITY = 0.25

# One colour per plotted cap, repeating after ten.
PLOT_COLORS = ((0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
               (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5),
               (0.74, 0.74, 0.13), (0.09, 0.75, 0.81))


def ensureVmtk():
    """Make `from vmtk import vtkvmtk` work, out of SlicerVMTK's modules if need be.

    A pip VMTK, where there is one, is left as it is. Under Slicer there is none and must not be
    -- see the module docstring -- so a module of that name is assembled from SlicerVMTK's, which
    carry the same classes under the same names. Done once; the second call finds it.
    """
    try:
        from vmtk import vtkvmtk  # noqa: F401
        return
    except ImportError:
        pass
    namespace = types.ModuleType("vmtk.vtkvmtk")
    found = []
    for name in VMTK_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        found.append(name)
        for attribute in dir(module):
            if attribute.startswith("vtkvmtk"):
                setattr(namespace, attribute, getattr(module, attribute))
    if not found:
        raise RuntimeError(_("VMTK is not available. Install the SlicerVMTK extension from the "
                             "Extensions Manager and restart Slicer."))
    package = types.ModuleType("vmtk")
    package.__path__ = []
    package.vtkvmtk = namespace
    sys.modules["vmtk"] = package
    sys.modules["vmtk.vtkvmtk"] = namespace


def pipEnsure(requirement, purpose, skipPackages=None):
    """Install `requirement` from PyPI into Slicer's Python if it is not there, asking first.

    Through `slicer.packaging.pip_ensure`, which asks before it downloads anything and installs
    nothing in a --testing run, so that a test never changes the Python it runs on. Declining is
    not raised here: the caller finds the package still missing and says how to install it later,
    which is more use than "declined". `purpose` finishes the question a Slicer older than
    `slicer.packaging` asks instead.
    """
    try:
        from slicer import packaging
        ensure = packaging.pip_ensure
    except (ImportError, AttributeError):
        ensure = None
    if ensure is not None:
        try:
            ensure(requirement, skip_packages=skipPackages, requester=_("SimVascular ROM Simulation"))
        except RuntimeError:
            pass
    elif slicer.util.confirmOkCancelDisplay(
            _("SimVascular ROM Simulation {purpose} with {requirement}, which is not installed. "
              "Install it from PyPI now?").format(purpose=purpose, requirement=requirement)):
        # Without slicer.packaging there is no skipping some of what a package asks for, only
        # all of it.
        slicer.util.pip_install(("--no-deps " if skipPackages else "") + requirement)
    importlib.invalidate_caches()


class SimVascularROM(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SimVascular ROM Simulation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SimVascular")]
        # Mesh Prep names the faces, and its folder is where svmeshcomplete is imported from.
        self.parent.dependencies = ["SimVascularMeshPrep"]
        self.parent.contributors = ["Cardiovascular Biomechanics Computation Lab (Stanford University)"]
        self.parent.helpText = _(
            "Set up a 0D (reduced-order) simulation of a vascular model whose faces are named "
            "in SimVascular Mesh Prep: one boundary condition per cap, centerlines from the "
            "inlet, an svZeroDSolver input file written by svromutils, the solver run, "
            "and the results shown per cap, over the cycle, and along the centerlines."
        )
        self.parent.acknowledgementText = _(
            "Developed in the Cardiovascular Biomechanics Computation Lab at Stanford "
            "University. The model is built by svromutils and solved by svZeroDSolver; "
            "the centerlines are VMTK's, through SlicerVMTK. The setting up is the svromsetup "
            "package beside this module, which runs outside Slicer as well."
        )


class SimVascularROMWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._updating = False
        # The selected mesh's faces, and the table of their conditions, which are shared with
        # SimVascular MultiPhysics: see SimVascularROMLib.BoundaryConditionsTable.
        self.namedFaces = NamedMeshFaces()
        self.conditionsTable = None
        self._inletFaceId = None
        # What the centerlines in the case folder were computed from, so that a run can tell
        # whether they are still the mesh's: changing the inlet, the folder or a name makes them
        # some other model's.
        self._centerlinesKey = None
        # What the solver files in the folder were written from, for the same reason: Run uses
        # them as they are while this still matches, which is what keeps an edit made to
        # solver_0d.json by hand between creating the files and running them.
        self._solverFilesKey = None
        self._faceResults = []
        self._resultsNote = ""
        # What the network view draws: the solver input and results it was read from, and the
        # centerlines a vessel clicked in it is shown on. Set with the results, and forgotten
        # with them.
        self._network = None
        self._config = None
        self._vesselResults = {}
        self._centerlines = None
        self.networkView = None
        self.networkDock = None
        # Set while the network view is selecting a cap in the table, so that the table selecting
        # it in the network view in turn does not start the round again.
        self._selectingFromNetwork = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimVascularROM.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimVascularROMLogic()
        self.conditionsTable = BoundaryConditionsTable(
            self.ui.facesTable, self.ui.valuesHintLabel, self.setStatus,
            changed=self.updateButtons, selectionChanged=self.onCapsSelected,
            startDirectory=lambda: self.ui.caseDirectoryPathLineEdit.currentPath)
        self.conditionsTable.emphasisNote = _("The source: the centerlines start here.")

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.openMeshPrepButton.connect("clicked(bool)", self.onOpenMeshPrep)
        self.ui.inletComboBox.connect("currentIndexChanged(int)", self.onInletChanged)
        self.ui.computeCenterlinesButton.connect("clicked(bool)", self.onComputeCenterlines)
        self.ui.createSolverFilesButton.connect("clicked(bool)", self.onCreateSolverFiles)
        self.ui.runButton.connect("clicked(bool)", self.onRun)
        self.ui.exportButton.connect("clicked(bool)", self.onExport)
        self.ui.networkButton.connect("clicked(bool)", self.onShowNetwork)
        self.ui.caseDirectoryPathLineEdit.connect("currentPathChanged(QString)",
                                                  self.onSimulationParameterChanged)
        for spinBox in (self.ui.densitySpinBox, self.ui.viscositySpinBox):
            spinBox.connect("valueChanged(double)", self.onSimulationParameterChanged)
        for spinBox in (self.ui.cyclesSpinBox, self.ui.pointsSpinBox):
            spinBox.connect("valueChanged(int)", self.onSimulationParameterChanged)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.restoreFromParameterNode()

    def enter(self):
        """Coming back, perhaps from Mesh Prep with the faces renamed, or from SimVascular
        MultiPhysics with the conditions changed: both are read again."""
        self.restoreFromParameterNode()

    def exit(self):
        clearHighlight()

    def cleanup(self):
        clearHighlight()
        self.logic.clearSelectedVessel()
        if self.networkDock is not None:
            slicer.util.mainWindow().removeDockWidget(self.networkDock)
            self.networkDock.deleteLater()
            self.networkDock = None
        self.removeObservers()

    def onSceneEndImport(self, caller=None, event=None):
        self.restoreFromParameterNode()

    def onSceneEndClose(self, caller=None, event=None):
        self._centerlinesKey = None
        self._solverFilesKey = None
        self._faceResults = []
        self._network = None
        self.onMeshChanged()

    # -- state that is saved: the conditions on the mesh, the rest on the scene ---
    def restoreFromParameterNode(self):
        node = self.logic.getParameterNode()
        self._updating = True
        try:
            directory = node.GetParameter(CASE_DIRECTORY_PARAMETER)
            if directory:
                self.ui.caseDirectoryPathLineEdit.currentPath = directory
            for parameter, spinBox, kind in (
                    (DENSITY_PARAMETER, self.ui.densitySpinBox, float),
                    (VISCOSITY_PARAMETER, self.ui.viscositySpinBox, float),
                    (CYCLES_PARAMETER, self.ui.cyclesSpinBox, int),
                    (POINTS_PARAMETER, self.ui.pointsSpinBox, int)):
                value = node.GetParameter(parameter)
                if value:
                    spinBox.value = kind(value)
            mesh = node.GetNodeReference(INPUT_MESH_REFERENCE)
            if mesh is not None:
                # Without its signal, which would measure the mesh once here and once below.
                wasBlocked = self.ui.inputMeshSelector.blockSignals(True)
                self.ui.inputMeshSelector.setCurrentNode(mesh)
                self.ui.inputMeshSelector.blockSignals(wasBlocked)
        finally:
            self._updating = False
        self.onMeshChanged()

    def saveToParameterNode(self):
        node = self.logic.getParameterNode()
        wasModifying = node.StartModify()
        node.SetParameter(CASE_DIRECTORY_PARAMETER, self.ui.caseDirectoryPathLineEdit.currentPath or "")
        node.SetParameter(DENSITY_PARAMETER, str(self.ui.densitySpinBox.value))
        node.SetParameter(VISCOSITY_PARAMETER, str(self.ui.viscositySpinBox.value))
        node.SetParameter(CYCLES_PARAMETER, str(self.ui.cyclesSpinBox.value))
        node.SetParameter(POINTS_PARAMETER, str(self.ui.pointsSpinBox.value))
        node.SetNodeReferenceID(INPUT_MESH_REFERENCE, self.ui.inputMeshSelector.currentNodeID or None)
        node.EndModify(wasModifying)

    def saveInlet(self):
        mesh = self.ui.inputMeshSelector.currentNode()
        if mesh is not None and self.conditionsTable.caps:
            mesh.SetAttribute(INLET_FACE_ID_ATTRIBUTE,
                              str(self._inletFaceId) if self._inletFaceId is not None else "")

    def onSimulationParameterChanged(self, *_args):
        if self._updating:
            return
        self.saveToParameterNode()
        self.updateButtons()

    # -- the mesh -------------------------------------------------------------------
    def onMeshChanged(self, _node=None):
        """Read the selected mesh's names, measure its caps, and bring back its conditions.

        The names are Mesh Prep's, read off the mesh every time rather than remembered, so that a
        rename there arrives here on coming back.
        """
        self._inletFaceId = None
        self._faceResults, self._resultsNote = [], ""
        self.ui.openMeshPrepButton.visible = False
        node = self.ui.inputMeshSelector.currentNode()
        if not self._updating:
            self.saveToParameterNode()
        if node is None or node.GetMesh() is None:
            self.ui.meshStatusLabel.text = ""
            self.showConditions(None)
            return
        try:
            named = self.namedFaces.read(node)
        except ValueError as error:
            self.ui.meshStatusLabel.text = str(error)
            self.showConditions(None)
            return
        self.ui.meshStatusLabel.text = self.namedFaces.describe()
        self.ui.openMeshPrepButton.visible = not named or bool(self.namedFaces.unnamed)
        if not named:
            self.showConditions(None)
            return
        self.showConditions(node)
        capIds = set(self.conditionsTable.capNames())
        try:
            saved = int(node.GetAttribute(INLET_FACE_ID_ATTRIBUTE) or "")
        except ValueError:
            saved = None
        self._inletFaceId = saved if saved in capIds else self.suggestedInlet()
        if not self.ui.caseDirectoryPathLineEdit.currentPath:
            self.ui.caseDirectoryPathLineEdit.currentPath = self.logic.suggestedCaseDirectory(node)
        self.populate()
        self.loadResultsFromCase()

    def showConditions(self, node):
        self.conditionsTable.setMesh(node, self.namedFaces if node is not None else None)
        self.populate()

    def suggestedInlet(self):
        """The largest cap given an inflow, or the largest cap: the inlet of most models.

        Any inflow would do -- the answer is the same, only the network is drawn from a different
        root -- but the largest is the trunk of the tree, and a tree drawn from its trunk is the
        one that reads as anatomy.
        """
        conditions = self.conditionsTable.conditions
        caps = self.conditionsTable.caps
        inflows = [(geometry.area, faceId) for faceId, _name, geometry in caps
                   if isinstance(conditions.get(faceId), bcs.Inflow)]
        candidates = inflows or [(geometry.area, faceId) for faceId, _name, geometry in caps]
        return max(candidates)[1] if candidates else None

    def onOpenMeshPrep(self):
        mesh = self.ui.inputMeshSelector.currentNode()
        slicer.util.selectModule("SimVascularMeshPrep")
        if mesh is not None:
            slicer.util.getModuleWidget("SimVascularMeshPrep").ui.inputMeshSelector.setCurrentNode(mesh)

    # -- the source, and the table it is emphasised in -----------------------------------
    def capName(self, faceId):
        return self.conditionsTable.capName(faceId)

    def populate(self):
        caps = self.conditionsTable.caps
        self._updating = True
        try:
            self.ui.inletComboBox.clear()
            for faceId, name, _geometry in caps:
                self.ui.inletComboBox.addItem(name, faceId)
            self.ui.inletComboBox.currentIndex = next(
                (row for row, (faceId, _name, _geometry) in enumerate(caps)
                 if faceId == self._inletFaceId), -1)
        finally:
            self._updating = False
        self.conditionsTable.emphasised = self._inletFaceId
        self.conditionsTable.populate()
        self.updateButtons()

    def onInletChanged(self, index):
        caps = self.conditionsTable.caps
        if self._updating or index < 0 or index >= len(caps):
            return
        self._inletFaceId = caps[index][0]
        self.saveInlet()
        self.populate()

    # -- centerlines and the run ---------------------------------------------------------
    def caseDirectory(self):
        directory = self.ui.caseDirectoryPathLineEdit.currentPath
        if not directory:
            raise RuntimeError(_("Choose an output folder to write the simulation into."))
        os.makedirs(directory, exist_ok=True)
        return directory

    def centerlinesKey(self):
        node = self.ui.inputMeshSelector.currentNode()
        return (node.GetID(), node.GetMesh().GetMTime(), self._inletFaceId,
                os.path.abspath(self.ui.caseDirectoryPathLineEdit.currentPath or ""),
                json.dumps(sorted(self.namedFaces.names.items())))

    def onComputeCenterlines(self):
        with slicer.util.tryWithErrorDisplay(_("Centerlines could not be computed."), waitCursor=True):
            self.computeCenterlines()

    def computeCenterlines(self):
        """Write the faces into the case, and trace and split the centerlines from the inlet."""
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or not self.conditionsTable.caps:
            raise RuntimeError(_("Select a mesh whose faces are named first."))
        if self._inletFaceId is None:
            raise RuntimeError(_("Choose the source the centerlines start from."))
        directory = self.caseDirectory()
        self.logic.importRomPackage()
        started = time.time()
        case.write_surfaces(node.GetMesh(), self.namedFaces.names, directory, self.namedFaces.arrayName)
        centerlines = case.compute_centerlines(directory, self._inletFaceId)
        self._centerlinesKey = self.centerlinesKey()
        self.logic.showCenterlines(node, centerlines.geometry)
        branches = len({int(branch) for branch in vtk_to_numpy(
            centerlines.geometry.GetPointData().GetArray("BranchId")) if branch >= 0})
        self.ui.centerlinesStatusLabel.text = _(
            "{branches} branches from {inlet} to {outlets} other caps, each centerline end paired "
            "with the cap it ends at ({seconds:.1f} s).").format(
            branches=branches, inlet=self.capName(self._inletFaceId),
            outlets=len(centerlines.outlet_names), seconds=time.time() - started)
        return centerlines

    def simulationParameters(self):
        return case.SimulationParameters(
            cardiac_cycles=int(self.ui.cyclesSpinBox.value),
            points_per_cycle=int(self.ui.pointsSpinBox.value),
            density=float(self.ui.densitySpinBox.value),
            viscosity=float(self.ui.viscositySpinBox.value),
            units=MESH_UNITS,
        )

    def onCreateSolverFiles(self):
        with slicer.util.tryWithErrorDisplay(_("The solver files could not be created."),
                                             waitCursor=True):
            self.setStatus(self.createSolverFiles())

    def checkConditions(self):
        capNames = self.conditionsTable.capNames()
        found = bcs.problems(self.conditionsTable.conditions, capNames, self._inletFaceId)
        if found:
            raise RuntimeError(" ".join(found))
        return capNames

    def solverFilesKey(self):
        node = self.ui.inputMeshSelector.currentNode()
        return (self.centerlinesKey(), bcs.to_json(self.conditionsTable.conditions), self.simulationParameters(),
                node.GetName())

    def createSolverFiles(self):
        """Write what svZeroDSolver reads into the output folder, without running it.

        The centerlines if they are not current, SimVascular's boundary condition files and
        `solver_0d.json` -- the case as the solver will see it, to look at or change before it is
        run. The centerlines are computed again only if they are not the current mesh's from the
        current inlet into the current folder, which is what makes trying another set of
        conditions cost a second rather than a centerline run.
        """
        node = self.ui.inputMeshSelector.currentNode()
        capNames = self.checkConditions()
        directory = self.caseDirectory()
        self.logic.importRomPackage()
        if self._centerlinesKey != self.centerlinesKey() or case.read_centerlines(directory) is None:
            self.computeCenterlines()
        byName = self.conditionsTable.conditionsByName()
        config = case.write_solver_input(directory, byName, self._inletFaceId,
                                         capNames[self._inletFaceId], self.simulationParameters(),
                                         sanitized(node.GetName()) or "model")
        self._solverFilesKey = self.solverFilesKey()
        with open(config) as handle:
            written = json.load(handle)
        return _("Wrote {config}: {vessels} vessels, {junctions} junctions and {conditions} "
                 "boundary conditions, with SimVascular's condition files beside it.").format(
            config=config, vessels=len(written.get("vessels", [])),
            junctions=len(written.get("junctions", [])),
            conditions=len(written.get("boundary_conditions", [])))

    def onRun(self):
        with slicer.util.tryWithErrorDisplay(_("The simulation could not be run."), waitCursor=True):
            self.runSimulation()

    def runSimulation(self):
        """Solve the solver files in the output folder, and show what they say.

        The files are written first only if they are not there, or the panel has changed since
        they were: otherwise they are run as they are, so that what Create solver files wrote --
        and anything changed in it by hand since -- is what is solved.
        """
        self.checkConditions()
        directory = self.caseDirectory()
        executable = self.logic.ensureSolver()
        config = os.path.join(directory, case.SOLVER_INPUT_NAME)
        if self._solverFilesKey != self.solverFilesKey() or not os.path.isfile(config):
            self.createSolverFiles()

        started = time.time()
        # With Slicer's own environment, not the one it started in: the solver is a command of
        # Slicer's Python, and that is the environment Slicer's Python needs.
        solver.run_solver(config, os.path.join(directory, case.RESULTS_NAME), executable)
        self.showResults(directory)
        flowIn = sum(face.mean_flow for face in self._faceResults if face.is_inflow)
        flowOut = sum(face.mean_flow for face in self._faceResults if not face.is_inflow)
        self.setStatus(_("Solved in {seconds:.1f} s: {inflow:.4g} mL/s in, {outflow:.4g} mL/s out "
                         "over the last cycle. Written to {directory}.").format(
            seconds=time.time() - started, inflow=flowIn, outflow=flowOut, directory=directory))

    def loadResultsFromCase(self):
        """Show the results a case folder already holds, if they are newer than its input.

        So that reopening a scene shows the last run again without running it. Older results than
        input are a run that failed after writing it, and are left alone.
        """
        directory = self.ui.caseDirectoryPathLineEdit.currentPath
        config = os.path.join(directory or "", case.SOLVER_INPUT_NAME)
        solved = os.path.join(directory or "", case.RESULTS_NAME)
        if not (directory and os.path.isfile(config) and os.path.isfile(solved)
                and os.path.getmtime(solved) >= os.path.getmtime(config)
                and self._inletFaceId is not None):
            self.populateResults()
            return
        try:
            self.showResults(directory)
            self._resultsNote = _("From the last run in {directory}.").format(directory=directory)
        except Exception as error:
            logging.warning("SimVascular ROM could not read the results in %s: %s", directory, error)
            self._faceResults = []
            self._network = None
        self.populateResults()

    def showResults(self, directory):
        node = self.ui.inputMeshSelector.currentNode()
        config = os.path.join(directory, case.SOLVER_INPUT_NAME)
        solved = solver.read_results(os.path.join(directory, case.RESULTS_NAME))
        self._faceResults = results.face_results(config, solved, self.capName(self._inletFaceId))
        self._resultsNote = ""
        with open(config) as handle:
            self._config = json.load(handle)
        self._vesselResults = solved
        self._network = network.network_from_config(self._config, self.capName(self._inletFaceId))
        centerlines = case.read_centerlines(directory)
        self._centerlines = centerlines
        self.logic.clearSelectedVessel()
        if centerlines is not None:
            coloured = results.centerline_results(centerlines, config, solved)
            self.logic.showResults(node, coloured, RESULTS_ARRAY_NAME)
        self.logic.writeResultsTable(node, self._faceResults)
        self.populateResults()
        self.plotSelected()

    # -- results -------------------------------------------------------------------------
    def populateResults(self):
        self.ui.exportButton.enabled = bool(self._faceResults)
        self.ui.networkButton.enabled = bool(self._faceResults) and self._network is not None
        if self._resultsNote:
            self.setStatus(self._resultsNote)
        if self.networkDock is not None and self.networkDock.visible:
            self.drawNetwork()

    def onCapsSelected(self, faceIds):
        """A selection in the conditions table: plot it, and select it in the network too."""
        self.plotSelected()
        if self._selectingFromNetwork or self.networkView is None or self._network is None:
            return
        self.logic.clearSelectedVessel()
        node = self._network.node_for_cap(self.capName(faceIds[0])) if len(faceIds) == 1 else None
        self.networkView.select(node.key if node is not None else None,
                                self.describeBlock(node.key) if node is not None else "")

    # -- the network -----------------------------------------------------------------------
    def onShowNetwork(self):
        with slicer.util.tryWithErrorDisplay(_("The network could not be shown.")):
            dock = self.ensureNetworkDock()
            dock.show()
            dock.raise_()
            self.drawNetwork()
            # Fitted again once the dock has a size, which it has only after it has been shown.
            qt.QTimer.singleShot(0, self.networkView.fit)

    def ensureNetworkDock(self):
        """The dock the network is drawn in, beside the views, made the first time it is asked for.

        A dock rather than a section of this panel: a Fontan's network is fifty blocks tall, which
        the panel has no room for, and a dock can be moved to the other side, or out of the
        window onto a second screen.
        """
        if self.networkDock is None:
            mainWindow = slicer.util.mainWindow()
            self.networkView = NetworkView(self.onNetworkBlockSelected)
            dock = qt.QDockWidget(_("0D network"), mainWindow)
            dock.objectName = NETWORK_DOCK_OBJECT_NAME
            dock.setWidget(self.networkView.widget)
            mainWindow.addDockWidget(qt.Qt.RightDockWidgetArea, dock)
            mainWindow.resizeDocks([dock], [460], qt.Qt.Horizontal)
            self.networkDock = dock
        return self.networkDock

    def drawNetwork(self):
        if self.networkView is None:
            return
        if self._network is None or not self._faceResults:
            self.networkView.clear(_("Run the simulation, and its network is drawn here."))
            return
        pressures = network.mean_pressures(self._network, self._vesselResults, self._faceResults)
        low, high = (min(pressures.values()), max(pressures.values())) if pressures else (0.0, 0.0)
        colours = {key: self.logic.pressureColour(value, low, high) for key, value in pressures.items()}
        tooltips = {key: self.describeBlock(key) for key in self._network.nodes}
        self.networkView.setNetwork(self._network, colours, tooltips, _(
            "Coloured by mean pressure, {low:.2f} to {high:.2f} mmHg, as the centerlines are. A "
            "vessel is labelled by its branch, and a letter for its segment if it has several: 25b "
            "is branch25_seg1. Scroll to zoom, drag to move, click a block for its values and "
            "results.").format(
                low=low, high=high))
        selected = self.conditionsTable.selectedFaceIds()
        if len(selected) == 1:
            self.onCapsSelected(selected)

    def describeBlock(self, key):
        return network.describe(self._network, key, self._vesselResults, self._faceResults)

    def onNetworkBlockSelected(self, key):
        """A block clicked in the network: say what it is, and show it in the plot and the anatomy.

        A condition is shown as its cap is when its row is selected -- by selecting that row, so
        that the table, the plot and the 3D view agree on what is selected. A vessel has no row:
        its two ends' pressures are plotted, and the stretch of centerline it was cut from is
        picked out on the results.
        """
        node = self._network.nodes.get(key) if (key and self._network is not None) else None
        self.networkView.setInfo(self.describeBlock(key) if node is not None else "")
        if node is not None and node.kind == network.BOUNDARY:
            self.logic.clearSelectedVessel()
            faceId = next((faceId for faceId, name, _geometry in self.conditionsTable.caps
                           if name == node.cap), None)
            if faceId is not None:
                self._selectingFromNetwork = True
                try:
                    self.ui.facesTable.selectRow(self.conditionsTable.rowOf(faceId))
                finally:
                    self._selectingFromNetwork = False
            return
        self._selectingFromNetwork = True
        try:
            self.ui.facesTable.clearSelection()
        finally:
            self._selectingFromNetwork = False
        clearHighlight()
        meshNode = self.ui.inputMeshSelector.currentNode()
        if node is None or node.kind != network.VESSEL or meshNode is None:
            self.logic.clearSelectedVessel()
            return
        result = self._vesselResults.get(node.vessel)
        if result is not None:
            table = self.logic.writeVesselTable(meshNode, result)
            self.logic.plotColumns(meshNode, table, [
                (f"{node.vessel} {end}", self.logic.vesselColumn(end)) for end, _field in VESSEL_COLUMNS],
                RESULTS_QUANTITY)
        if self._centerlines is not None:
            self.logic.showSelectedVessel(
                meshNode, results.vessel_lines(self._centerlines, self._config, node.vessel))

    def plotSelected(self):
        """Plot the caps selected in the boundary conditions table, or the inlet if none are.

        The table the conditions are set in is the one results are asked of, because it is the
        one with a row per cap already, and selecting a row there already shows that cap in the
        3D view: one selection says which cap, in the anatomy and in the plot.
        """
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or not self._faceResults:
            return
        solved = {face.name for face in self._faceResults}
        names = [self.capName(faceId) for faceId in self.conditionsTable.selectedFaceIds()
                 if self.capName(faceId) in solved]
        if not names and self.capName(self._inletFaceId) in solved:
            names = [self.capName(self._inletFaceId)]
        if names:
            self.logic.plotFaces(node, names, RESULTS_QUANTITY)

    def onExport(self):
        path = self.chooseExportFile()
        if not path:
            return
        with slicer.util.tryWithErrorDisplay(_("Could not export the results."), waitCursor=True):
            self.setStatus(self.exportResults(path))

    def chooseExportFile(self):
        """Where the results are to be written, or "" if nowhere was chosen.

        Its own method, as chooseWaveformFile is, so that a test can answer it. Suggested in the
        output folder, beside the solver's own per-segment results, under a name that says it is
        the per-cap one; not after the model, which the output folder is already named for.
        """
        suggested = os.path.join(self.ui.caseDirectoryPathLineEdit.currentPath or "",
                                 case.CAP_RESULTS_NAME)
        return qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(), _("Export results"), suggested,
            _("CSV files (*.csv);;All files (*)"))

    def exportResults(self, path):
        written = results.write_face_results_csv(path, self._faceResults)
        return _("Wrote {count} caps' flow and pressure over the last cycle to {path}.").format(
            count=len(self._faceResults), path=written)

    # -- panel state -----------------------------------------------------------------------
    def updateButtons(self):
        if self.conditionsTable is None:
            return
        caps = self.conditionsTable.caps
        found = (bcs.problems(self.conditionsTable.conditions, self.conditionsTable.capNames(),
                              self._inletFaceId) if caps else [])
        ready = bool(caps) and not found and bool(self.ui.caseDirectoryPathLineEdit.currentPath)
        self.ui.computeCenterlinesButton.enabled = bool(caps) and self._inletFaceId is not None
        self.ui.createSolverFilesButton.enabled = ready
        self.ui.runButton.enabled = ready
        notReady = " ".join(found) or _("Choose an output folder.")
        self.ui.createSolverFilesButton.toolTip = (
            _("Write the boundary condition files and svZeroDSolver's solver_0d.json into the "
              "output folder, computing the centerlines first if they are not current, without "
              "running anything. Run simulation solves them as they are, so they can be looked "
              "at or changed by hand in between.") if ready else notReady)
        self.ui.runButton.toolTip = (
            _("Run svZeroDSolver on the solver files and show the results. The files are created "
              "first if they are not there, or the panel has changed since they were.")
            if ready else notReady)
        if caps and not self._faceResults:
            self.setStatus(" ".join(found) if found else _("Ready to run."), warning=bool(found))

    def setStatus(self, text, warning=False):
        self.ui.statusLabel.text = text
        self.ui.statusLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""


class SimVascularROMLogic(ScriptedLoadableModuleLogic):
    """What touches the scene: the names on the mesh, and the nodes a run leaves beside it."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def importRomPackage():
        """Make `svromutils` importable: VMTK from SlicerVMTK, the package from PyPI.

        Installed the first time it is needed, asked to skip what svromutils declares it depends
        on (see ROM_PACKAGE_SKIPPED_DEPENDENCIES): it is installed without them, and they are
        recorded as skipped, so that nothing pip does later goes back and installs them over
        Slicer's own.
        """
        ensureVmtk()
        try:
            import svromutils  # noqa: F401
            return
        except ImportError:
            pass
        pipEnsure(ROM_PACKAGE_REQUIREMENT, _("builds its model"), ROM_PACKAGE_SKIPPED_DEPENDENCIES)
        try:
            import svromutils  # noqa: F401
        except ImportError:
            raise RuntimeError(_(
                "svromutils is not installed. It is installed from PyPI the first time it is "
                "needed, when asked; to install it by hand, run "
                "slicer.util.pip_install('--no-deps svromutils') in the Python console.")) from None

    @staticmethod
    def ensureSolver():
        """svzerod's `svzerodsolver` beside Slicer's own Python, installed from PyPI if it is not.

        Only ever that one, never one found on the PATH -- a build of svZeroDSolver's own, or a
        conda environment's -- because it is run with Slicer's environment, which is what Slicer's
        Python needs, and a command of any other Python run with it starts on Slicer's instead.
        """
        executable = solver.installed_solver()
        if executable is None:
            pipEnsure(solver.SOLVER_PACKAGE, _("solves its model"))
            executable = solver.installed_solver()
        if executable is None:
            raise RuntimeError(_(
                "svzerod (svZeroDSolver) is not installed. It is installed from PyPI the first "
                "time a simulation is run, when asked; to install it by hand, run "
                "slicer.util.pip_install('svzerod') in the Python console."))
        return executable

    @staticmethod
    def suggestedCaseDirectory(meshNode):
        """`rom` beside the saved scene, once the scene has been saved; nowhere until then."""
        url = slicer.mrmlScene.GetURL()
        directory = os.path.dirname(url) if url else ""
        if not os.path.isdir(directory):
            return ""
        return os.path.join(directory, "rom", sanitized(meshNode.GetName()) or "model")

    # -- nodes a run leaves --------------------------------------------------------------
    @staticmethod
    def referencedNode(meshNode, role, className, name):
        """The node the mesh references in `role`, made if it has none, named after the mesh."""
        node = meshNode.GetNodeReference(role)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(className, f"{meshNode.GetName()} {name}")
            meshNode.SetNodeReferenceID(role, node.GetID())
        return node

    def showCenterlines(self, meshNode, centerlines):
        node = self.referencedNode(meshNode, CENTERLINES_REFERENCE, "vtkMRMLModelNode",
                                   "ROM centerlines")
        node.SetAndObserveMesh(centerlines)
        node.SetAndObserveTransformNodeID(meshNode.GetTransformNodeID())
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetColor(0.1, 0.5, 1.0)
        display.SetLineWidth(3)
        display.SetScalarVisibility(False)
        display.SetVisibility(True)
        # Faded as for results: the centerlines run inside the vessel, so behind an opaque mesh
        # nothing on screen changes when they have been computed.
        meshDisplay = meshNode.GetDisplayNode()
        if meshDisplay is not None and meshDisplay.GetOpacity() > RESULTS_MESH_OPACITY:
            meshDisplay.SetOpacity(RESULTS_MESH_OPACITY)
        results = meshNode.GetNodeReference(RESULTS_MODEL_REFERENCE)
        if results is not None and results.GetDisplayNode() is not None:
            results.GetDisplayNode().SetVisibility(False)
        return node

    def showResults(self, meshNode, coloured, arrayName):
        """The centerlines as tubes of the inscribed sphere radius, coloured by `arrayName`.

        Tubes rather than lines because a line is one pixel wide whatever it carries, and the
        colour is the point; the inscribed sphere radius is what VMTK traced them by, so the tube
        sits inside the vessel the way the vessel does. The mesh is faded so that it can be seen
        through to them.
        """
        tube = vtk.vtkTubeFilter()
        radius = coloured.GetPointData().GetArray("MaximumInscribedSphereRadius")
        withRadius = vtk.vtkPolyData()
        withRadius.ShallowCopy(coloured)
        if radius is not None:
            withRadius.GetPointData().SetActiveScalars("MaximumInscribedSphereRadius")
            tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        else:
            tube.SetRadius(0.5)
        tube.SetInputData(withRadius)
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        tube.Update()

        node = self.referencedNode(meshNode, RESULTS_MODEL_REFERENCE, "vtkMRMLModelNode", "ROM results")
        node.SetAndObserveMesh(tube.GetOutput())
        node.SetAndObserveTransformNodeID(meshNode.GetTransformNodeID())
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetVisibility(True)
        self.setResultsQuantity(meshNode, arrayName)

        centerlines = meshNode.GetNodeReference(CENTERLINES_REFERENCE)
        if centerlines is not None and centerlines.GetDisplayNode() is not None:
            centerlines.GetDisplayNode().SetVisibility(False)
        meshDisplay = meshNode.GetDisplayNode()
        if meshDisplay is not None and meshDisplay.GetOpacity() > RESULTS_MESH_OPACITY:
            meshDisplay.SetOpacity(RESULTS_MESH_OPACITY)
        return node

    @staticmethod
    def setResultsQuantity(meshNode, arrayName):
        node = meshNode.GetNodeReference(RESULTS_MODEL_REFERENCE)
        if node is None or node.GetDisplayNode() is None:
            return
        display = node.GetDisplayNode()
        display.SetActiveScalarName(arrayName)
        display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        for colorNodeId in RESULTS_COLOR_NODES:
            if slicer.mrmlScene.GetNodeByID(colorNodeId) is not None:
                display.SetAndObserveColorNodeID(colorNodeId)
                break
        display.SetScalarRangeFlag(display.UseDataScalarRange)
        display.SetScalarVisibility(True)
        # A legend, where this Slicer has them: a colour without its scale says nothing.
        try:
            colors = slicer.modules.colors.logic()
            legend = colors.GetColorLegendDisplayNode(node) or colors.AddDefaultColorLegendDisplayNode(node)
            legend.SetTitleText(arrayName)
            # Two decimals: a model's pressures often span a fraction of a mmHg, and a legend
            # reading 12.1, 12.1, 12.0 says nothing about which end is which.
            legend.SetLabelFormat("%.2f")
            legend.SetVisibility(True)
        except Exception:
            logging.debug("SimVascular ROM could not show a colour legend.", exc_info=True)

    def writeResultsTable(self, meshNode, faceResults):
        """Every cap's pressure and flow over the last cycle, one column each, for plotting."""
        node = self.referencedNode(meshNode, RESULTS_TABLE_REFERENCE, "vtkMRMLTableNode",
                                   "ROM results table")
        table = vtk.vtkTable()
        if faceResults:
            columns = [(TIME_COLUMN, faceResults[0].time)]
            for face in faceResults:
                columns.append((f"{face.name} pressure [mmHg]", face.pressure / bcs.MMHG))
                columns.append((f"{face.name} flow [mL/s]", face.flow))
            for name, values in columns:
                array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
                array.SetName(name)
                table.AddColumn(array)
        node.SetAndObserveTable(table)
        return node

    @staticmethod
    def vesselColumn(end):
        return f"{end} pressure [mmHg]"

    def writeVesselTable(self, meshNode, result):
        """One vessel's pressures and flows at its two ends, over the last cycle, for plotting."""
        node = self.referencedNode(meshNode, VESSEL_TABLE_REFERENCE, "vtkMRMLTableNode",
                                   "ROM vessel results")
        table = vtk.vtkTable()
        columns = [(TIME_COLUMN, result.time)]
        for end, field in VESSEL_COLUMNS:
            columns.append((self.vesselColumn(end), getattr(result, field) / bcs.MMHG))
        columns += [("inlet flow [mL/s]", result.flow_in), ("outlet flow [mL/s]", result.flow_out)]
        for name, values in columns:
            array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
            array.SetName(name)
            table.AddColumn(array)
        node.SetAndObserveTable(table)
        return node

    @staticmethod
    def selectedVesselNode():
        """The selected vessel's node, which being hidden is not found by name lookups."""
        for index in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = slicer.mrmlScene.GetNthNodeByClass(index, "vtkMRMLModelNode")
            if node.GetName() == SELECTED_VESSEL_NODE_NAME:
                return node
        return None

    def showSelectedVessel(self, meshNode, lines):
        """Pick out a vessel on the anatomy: its stretch of centerline, as a sleeve round the
        results' tubes -- a little wider than the inscribed sphere they are drawn at, and see-through,
        so that the pressure colour inside it still shows."""
        radius = lines.GetPointData().GetArray("MaximumInscribedSphereRadius")
        scaled = vtk.vtkPolyData()
        scaled.ShallowCopy(lines)
        tube = vtk.vtkTubeFilter()
        if radius is not None and radius.GetNumberOfTuples():
            array = numpy_to_vtk(vtk_to_numpy(radius) * SELECTED_VESSEL_RADIUS_FACTOR, deep=True)
            array.SetName("SleeveRadius")
            scaled.GetPointData().AddArray(array)
            scaled.GetPointData().SetActiveScalars("SleeveRadius")
            tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        else:
            tube.SetRadius(1.0)
        tube.SetInputData(scaled)
        tube.SetNumberOfSides(20)
        tube.CappingOn()
        tube.Update()

        node = self.selectedVesselNode()
        if node is None:
            node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelNode")
            node.SetName(SELECTED_VESSEL_NODE_NAME)
            node.SetSaveWithScene(False)
            node.SetHideFromEditors(True)
            node = slicer.mrmlScene.AddNode(node)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSaveWithScene(False)
            node.GetDisplayNode().SetHideFromEditors(True)
        node.SetAndObserveMesh(tube.GetOutput())
        node.SetAndObserveTransformNodeID(meshNode.GetTransformNodeID())
        display = node.GetDisplayNode()
        # Mesh Prep's face highlight, which a selected cap is shown in: one colour for a selection,
        # whichever kind of block it is. Imported here, as the conditions table imports Mesh Prep's
        # highlighting, so that loading this module does not depend on Mesh Prep's loading first.
        from SimVascularMeshPrep import HIGHLIGHT_COLOR
        display.SetColor(*HIGHLIGHT_COLOR)
        display.SetOpacity(SELECTED_VESSEL_OPACITY)
        display.SetScalarVisibility(False)
        display.SetVisibility(True)
        return node

    def clearSelectedVessel(self):
        node = self.selectedVesselNode()
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def pressureColour(value, low, high):
        """The colour the results' tubes give `value` over `low`..`high`, as (r, g, b) in 0..1,
        so that a block in the network and its stretch of centerline are the same colour."""
        colourNode = next((slicer.mrmlScene.GetNodeByID(nodeId) for nodeId in RESULTS_COLOR_NODES
                           if slicer.mrmlScene.GetNodeByID(nodeId) is not None), None)
        if colourNode is None:
            return None
        fraction = 0.5 if high <= low else min(max((value - low) / (high - low), 0.0), 1.0)
        rgba = [0.0, 0.0, 0.0, 0.0]
        colourNode.GetColor(int(round(fraction * (colourNode.GetNumberOfColors() - 1))), rgba)
        return tuple(rgba[:3])

    def plotFaces(self, meshNode, names, quantity):
        """Plot `quantity` over the last cycle for the named caps, and put the plot in the layout."""
        tableNode = meshNode.GetNodeReference(RESULTS_TABLE_REFERENCE)
        if tableNode is None:
            return None
        unit = "mmHg" if quantity == "Pressure" else "mL/s"
        return self.plotColumns(meshNode, tableNode,
                                [(name, f"{name} {quantity.lower()} [{unit}]") for name in names], quantity)

    def plotColumns(self, meshNode, tableNode, series, quantity):
        """Plot (legend, column) pairs of `tableNode` over the last cycle, in the mesh's plot."""
        chart = self.referencedNode(meshNode, RESULTS_CHART_REFERENCE, "vtkMRMLPlotChartNode",
                                    "ROM plot")
        unit = "mmHg" if quantity == "Pressure" else "mL/s"
        chart.SetTitle(f"{quantity} over the last cycle")
        chart.SetXAxisTitle(TIME_COLUMN)
        chart.SetYAxisTitle(f"{quantity.lower()} [{unit}]")
        chart.RemoveAllPlotSeriesNodeIDs()
        for index, (name, column) in enumerate(series):
            plotted = self.plotSeries(tableNode, column)
            if plotted is None:
                # Found again by the table and column it plots, which is what makes it this cap's
                # or this vessel's end; named for what the plot's legend is to show.
                plotted = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", name)
                plotted.SetAttribute(PLOT_COLUMN_ATTRIBUTE, column)
                plotted.SetAndObserveTableNodeID(tableNode.GetID())
            # Named again every time: the vessel table's columns are every vessel's in turn.
            plotted.SetName(name)
            plotted.SetXColumnName(TIME_COLUMN)
            plotted.SetYColumnName(column)
            plotted.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            plotted.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
            plotted.SetColor(*PLOT_COLORS[index % len(PLOT_COLORS)])
            chart.AddAndObservePlotSeriesNodeID(plotted.GetID())
        slicer.modules.plots.logic().ShowChartInLayout(chart)
        # A maximized view -- a scene can be saved with its 3D view maximized -- covers the plot
        # the line above switched the layout to, and then asking for a plot shows nothing at all.
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if layoutNode is not None and layoutNode.GetNumberOfMaximizedViewNodes():
            layoutNode.RemoveAllMaximizedViewNodes()
        return chart

    @staticmethod
    def plotSeries(tableNode, column):
        """The series already plotting `column` of `tableNode`, or None."""
        for series in slicer.util.getNodesByClass("vtkMRMLPlotSeriesNode"):
            if (series.GetAttribute(PLOT_COLUMN_ATTRIBUTE) == column
                    and series.GetTableNodeID() == tableNode.GetID()):
                return series
        return None


class SimVascularROMTest(ScriptedLoadableModuleTest):
    """Runs under Slicer; the package's own tests run without it."""

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.mrmlScene.SetURL("")

    def runTest(self):
        self.setUp()
        self.test_conditionsAreKeptOnTheMeshTheyAreFor()
        self.setUp()
        self.test_doubleClickingAnInflowLoadsItsWaveform()
        self.setUp()
        self.test_aMeshWithoutNamesIsSentToMeshPrep()
        self.setUp()
        self.test_aWholeCaseOnAY()

    @staticmethod
    def clickBlock(view, key):
        """A click on a block's middle, sent through Qt as a person's is, press and release."""
        view.view.ensureVisible(view.items[key])
        slicer.app.processEvents()
        point = view.view.mapFromScene(view.items[key].sceneBoundingRect().center())
        onScreen = view.view.viewport().mapToGlobal(point)
        for kind, buttons in ((qt.QEvent.MouseButtonPress, qt.Qt.LeftButton),
                              (qt.QEvent.MouseButtonRelease, qt.Qt.NoButton)):
            qt.QApplication.sendEvent(view.view.viewport(), qt.QMouseEvent(
                kind, qt.QPointF(point), qt.QPointF(onScreen), qt.Qt.LeftButton, buttons, qt.Qt.NoModifier))
            slicer.app.processEvents()

    def namedY(self, name="Y"):
        from svromsetup import testing

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObserveMesh(testing.y_surface())
        node.CreateDefaultDisplayNodes()
        node.SetAttribute(FACE_NAMES_ATTRIBUTE, json.dumps(
            {str(faceId): faceName for faceId, faceName in testing.NAMES.items()}))
        return node

    def widget(self):
        widget = slicer.util.getModuleWidget("SimVascularROM")
        # The widget outlives the scene, so what an earlier test left on it is reset here.
        widget.ui.caseDirectoryPathLineEdit.currentPath = ""
        return widget

    def typeCondition(self, widget, faceId, kind, text):
        row = widget.conditionsTable.rowOf(faceId)
        widget.ui.facesTable.cellWidget(row, TYPE_COLUMN).currentIndex = KIND_CHOICES.index(kind)
        widget.ui.facesTable.item(row, VALUES_COLUMN).setText(text)

    def test_conditionsAreKeptOnTheMeshTheyAreFor(self):
        """Two anatomies with the same face ids keep their own conditions, and get them back."""
        from svromsetup import testing

        widget = self.widget()
        first, second = self.namedY("first"), self.namedY("second")
        widget.ui.inputMeshSelector.setCurrentNode(first)
        self.assertEqual([faceId for faceId, _name, _geometry in widget.conditionsTable.caps],
                         [testing.INLET_ID, testing.RIGHT_ID, testing.LEFT_ID])
        self.assertEqual(widget._inletFaceId, testing.INLET_ID, "the largest cap is the inlet")
        self.typeCondition(widget, testing.RIGHT_ID, bcs.RESISTANCE_KIND, "1500")
        self.assertEqual(widget.conditionsTable.conditions[testing.RIGHT_ID], bcs.Resistance(1500.0))

        widget.ui.inputMeshSelector.setCurrentNode(second)
        self.assertEqual(widget.conditionsTable.conditions, {}, "a condition set on one anatomy is not on another")
        widget.ui.inputMeshSelector.setCurrentNode(first)
        self.assertEqual(widget.conditionsTable.conditions, {testing.RIGHT_ID: bcs.Resistance(1500.0)})
        self.assertEqual(bcs.from_json(first.GetAttribute(BOUNDARY_CONDITIONS_ATTRIBUTE)),
                         {testing.RIGHT_ID: bcs.Resistance(1500.0)})

        # What cannot be read stays on screen, in red, and is not a condition.
        self.typeCondition(widget, testing.LEFT_ID, bcs.RCR_KIND, "1, 2")
        self.assertNotIn(testing.LEFT_ID, widget.conditionsTable.conditions)
        item = widget.ui.facesTable.item(widget.conditionsTable.rowOf(testing.LEFT_ID), VALUES_COLUMN)
        self.assertEqual(item.text(), "1, 2")
        self.assertEqual(item.foreground().color().red(), INVALID_VALUES_COLOR.red())
        self.assertFalse(widget.ui.runButton.enabled)
        self.delayDisplay("Conditions kept per mesh")

    def test_doubleClickingAnInflowLoadsItsWaveform(self):
        """On an inflow the values are a file to load; on anything else, values to type.

        The file dialog is answered by replacing the method that opens it, which is the one thing
        here that would wait for a person.
        """
        import tempfile

        from svromsetup import testing

        widget = self.widget()
        widget.ui.inputMeshSelector.setCurrentNode(self.namedY())
        table = widget.ui.facesTable
        # Written as svMultiPhysics writes an inflow, negative, so that turning it round is
        # checked too.
        path = os.path.join(tempfile.mkdtemp(), "inlet.flow")
        with open(path, "w") as handle:
            handle.write("0.0 -5.0\n0.4 -15.0\n0.8 -5.0\n")
        asked = []
        widget.conditionsTable.chooseWaveformFile = lambda faceId: asked.append(faceId) or path
        try:
            self.typeCondition(widget, testing.INLET_ID, bcs.INFLOW, "")
            widget.conditionsTable.onCellDoubleClicked(widget.conditionsTable.rowOf(testing.INLET_ID), VALUES_COLUMN)
            self.assertEqual(asked, [testing.INLET_ID])
            self.assertEqual(widget.conditionsTable.conditions[testing.INLET_ID],
                             bcs.Inflow((0.0, 0.4, 0.8), (5.0, 15.0, 5.0)))
            self.assertIn("turned round", widget.ui.statusLabel.text)
            self.assertEqual(table.item(widget.conditionsTable.rowOf(testing.INLET_ID), VALUES_COLUMN).text(),
                             os.path.realpath(path), "the cell names the file the waveform came from")

            # Anywhere else a double-click opens the values for typing and asks for no file.
            self.typeCondition(widget, testing.RIGHT_ID, bcs.RCR_KIND, "100, 1e-4, 1000")
            widget.conditionsTable.onCellDoubleClicked(widget.conditionsTable.rowOf(testing.RIGHT_ID), VALUES_COLUMN)
            self.assertEqual(asked, [testing.INLET_ID])
            self.assertEqual(table.state(), qt.QAbstractItemView.EditingState)
            # Closed with Escape, as a person would close it. Not with reset(), which closes
            # editors by deleting every widget in the view's cells -- the condition combo boxes
            # with them.
            for editor in table.viewport().findChildren("QLineEdit"):
                qt.QApplication.sendEvent(editor, qt.QKeyEvent(qt.QEvent.KeyPress, qt.Qt.Key_Escape,
                                                               qt.Qt.NoModifier))
            self.assertNotEqual(table.state(), qt.QAbstractItemView.EditingState)
            self.assertIsNotNone(table.cellWidget(widget.conditionsTable.rowOf(testing.RIGHT_ID), TYPE_COLUMN))
            # And a cap with no kind yet is left alone.
            widget.conditionsTable.onCellDoubleClicked(widget.conditionsTable.rowOf(testing.LEFT_ID), VALUES_COLUMN)
            self.assertEqual(asked, [testing.INLET_ID])
        finally:
            del widget.conditionsTable.chooseWaveformFile
        self.delayDisplay("Double-clicking an inflow loads its waveform")

    def test_aMeshWithoutNamesIsSentToMeshPrep(self):
        from svmeshcomplete import testing

        widget = self.widget()
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "unnamed")
        node.SetAndObserveMesh(testing.cube_mesh())
        widget.ui.inputMeshSelector.setCurrentNode(node)
        self.assertEqual(widget.conditionsTable.caps, [])
        self.assertFalse(widget.ui.openMeshPrepButton.isHidden())
        self.assertIn("Mesh Prep", widget.ui.meshStatusLabel.text)
        self.delayDisplay("An unnamed mesh is sent to Mesh Prep")

    def test_aWholeCaseOnAY(self):
        """Through the package and the solver, where both are on this machine."""
        import tempfile

        from svromsetup import testing

        widget = self.widget()
        try:
            # A --testing run installs nothing, so this runs where svromutils and svzerod are
            # installed into Slicer already.
            widget.logic.importRomPackage()
            widget.logic.ensureSolver()
        except RuntimeError as error:
            self.delayDisplay(f"Skipped: {error}")
            return
        node = self.namedY()
        widget.ui.inputMeshSelector.setCurrentNode(node)
        widget.ui.caseDirectoryPathLineEdit.currentPath = tempfile.mkdtemp()
        self.typeCondition(widget, testing.INLET_ID, bcs.INFLOW, "10")
        self.typeCondition(widget, testing.RIGHT_ID, bcs.RESISTANCE_KIND, "2000")
        self.typeCondition(widget, testing.LEFT_ID, bcs.INFLOW, "3")
        self.assertTrue(widget.ui.runButton.enabled)
        widget.ui.cyclesSpinBox.value = 3
        widget.ui.pointsSpinBox.value = 20
        widget.runSimulation()

        faces = {face.name: face for face in widget._faceResults}
        self.assertAlmostEqual(faces["cap_right"].mean_flow, 13.0, places=4)
        self.assertIsNotNone(node.GetNodeReference(RESULTS_MODEL_REFERENCE))
        table = node.GetNodeReference(RESULTS_TABLE_REFERENCE).GetTable()
        self.assertIsNotNone(table.GetColumnByName("cap_right pressure [mmHg]"))

        # Selecting a cap in the conditions table plots it, and the export holds every cap.
        widget.ui.facesTable.selectRow(widget.conditionsTable.rowOf(testing.RIGHT_ID))
        chart = node.GetNodeReference(RESULTS_CHART_REFERENCE)
        self.assertEqual(chart.GetNthPlotSeriesNode(0).GetYColumnName(), "cap_right pressure [mmHg]")
        self.assertTrue(widget.ui.exportButton.enabled)
        directory = widget.ui.caseDirectoryPathLineEdit.currentPath
        self.assertTrue(os.path.isfile(os.path.join(directory, case.RESULTS_NAME)))
        exported = os.path.join(directory, case.CAP_RESULTS_NAME)
        widget.exportResults(exported)
        with open(exported) as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header[0], "time [s]")
        self.assertIn("cap_left flow [mL/s]", header)

        # The network: every block drawn; a cap's block selects the cap's row, which is what
        # plots it and shows it on the anatomy; a vessel's plots its two ends and picks it out on
        # the centerlines; and the table selecting a cap selects its block.
        widget.onShowNetwork()
        view = widget.networkView
        self.assertEqual(set(view.items), set(widget._network.nodes))
        right = widget._network.node_for_cap("cap_right")
        self.clickBlock(view, right.key)
        self.assertEqual(widget.conditionsTable.selectedFaceIds(), [testing.RIGHT_ID])
        self.assertIn("cap_right: Resistance", view.infoLabel.text)
        (vessel,) = widget._network.parents(right.key)
        self.clickBlock(view, vessel)
        self.assertEqual(widget.conditionsTable.selectedFaceIds(), [])
        vesselName = widget._network.nodes[vessel].vessel
        self.assertEqual(chart.GetNthPlotSeriesNode(0).GetName(), f"{vesselName} inlet")
        sleeve = widget.logic.selectedVesselNode()
        self.assertIsNotNone(sleeve)
        self.assertGreater(sleeve.GetMesh().GetNumberOfPoints(), 0)
        widget.ui.facesTable.selectRow(widget.conditionsTable.rowOf(testing.LEFT_ID))
        self.assertEqual(view.selectedKey(), widget._network.node_for_cap("cap_left").key)
        self.assertIsNone(widget.logic.selectedVesselNode(), "a cap selected clears the vessel")

        # Created solver files are run as they are: a resistance doubled by hand in
        # solver_0d.json between creating and running is the one solved. Changing a condition in
        # the panel afterwards makes Run write them again, from the panel.
        def pressureAtRight():
            return {face.name: face for face in widget._faceResults}["cap_right"].mean_pressure_mmhg
        before = pressureAtRight()
        self.assertTrue(widget.ui.createSolverFilesButton.enabled)
        widget.createSolverFiles()
        config = os.path.join(widget.ui.caseDirectoryPathLineEdit.currentPath, case.SOLVER_INPUT_NAME)
        with open(config) as handle:
            edited = json.load(handle)
        for condition in edited["boundary_conditions"]:
            if condition["bc_name"] == "RESISTANCE_cap_right":
                condition["bc_values"]["R"] *= 2
        with open(config, "w") as handle:
            json.dump(edited, handle)
        widget.runSimulation()
        self.assertAlmostEqual(pressureAtRight(), 2 * before, places=2)
        self.typeCondition(widget, testing.RIGHT_ID, bcs.RESISTANCE_KIND, "2000")
        self.typeCondition(widget, testing.LEFT_ID, bcs.INFLOW, "3.5")
        widget.runSimulation()
        self.assertAlmostEqual(pressureAtRight(), before * 13.5 / 13.0, places=2)
        self.delayDisplay("A whole case on a Y")
