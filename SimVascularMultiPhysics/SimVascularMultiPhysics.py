"""Write the svMultiPhysics case for a labelled vascular mesh: the mesh, the waveforms, `solver.xml`.

svMultiPhysics is run on a folder: a `solver.xml` naming every face of the mesh and the condition
on each, the mesh-complete folder those faces are files in, and the waveform of each inflow in a
format of its own. Writing that by hand is where a case goes wrong without a word -- a wall face
left out of the file is solved as a hole, a scale factor left out makes a mesh in mm ten times too
large, an inflow written positive draws flow out of the model, a number with a typo in it is read
as its longest numeric prefix. This panel writes it from what the scene already knows.

The mesh is one SimVascular Mesh Prep has named, and its names are read off it, never typed again
here. The conditions are the ones SimVascular ROM Simulation sets for the same mesh -- one set per
mesh, kept on the mesh node, edited in either panel -- so a 0D model tuned there is the 3D case
written here, with nothing copied between them to go stale. The table they are set in is the 0D
panel's own, shared rather than copied: see SimVascularROMLib.BoundaryConditionsTable.

Everything else in the file is a setting with a default, and every one can be changed. They are
one table, `svmpsetup.settings`, and this panel draws a field for each row of it rather than a
field of its own for each: a default changed there is changed here, and there is no second
default anywhere for the two to disagree about.

## Where the work is

In `svmpsetup`, the package beside this file, which imports nothing from Slicer. This file is the
MRML adapter: it reads the selected node, calls the package, and says what it wrote. A case
written from a terminal calls the same functions.

## Rigid-wall CFD only

The flow in the lumen, with walls that do not move: the one kind of simulation this panel writes,
for a class running svMultiPhysics as it comes. Its linear solver is fixed for that, Navier-Stokes
through svMultiPhysics' own linear algebra, and the panel says so rather than offering a choice.

## A few steps here before the cluster

The panel writes the case and does not run it, but it will run a few steps of it, so that what
would stop the full run is found in minutes here rather than after a queue on a cluster. The test,
and how the solver's output is read, are `svmpsetup.trial`'s; here they are run as a process of
the application's own, with the environment Slicer was started with -- not the one its launcher
made, whose library folders, Slicer's VTK among them, are no business of a program built against
its own.
"""

import json
import logging
import os
import sys
import time

# This module's own folder first, so that `svmpsetup` is the package beside it. With the extension
# installed, Slicer's launcher puts the installed copy's folder on the Python path before any module
# is found, so a module loaded from a checkout -- through the module paths, as a developer's is --
# would otherwise import the installed package, of whatever version that is, without a word; and
# fail on anything the checkout added. Installed, the two folders are one and nothing changes.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path[:1] != [_MODULE_DIR]:
    sys.path.insert(0, _MODULE_DIR)

import ctk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

import vtk

from svmeshcomplete.face_table import sanitized
from svmpsetup import case, settings, trial
from svromsetup import boundary_conditions as bcs

# The faces and the conditions table, shared with SimVascular ROM Simulation.
from SimVascularROMLib.BoundaryConditionsTable import (  # noqa: F401 (the tests use these)
    BOUNDARY_CONDITIONS_ATTRIBUTE,
    FACE_NAMES_ATTRIBUTE,
    KIND_CHOICES,
    TYPE_COLUMN,
    VALUES_COLUMN,
    BoundaryConditionsTable,
    NamedMeshFaces,
    clearHighlight,
)

# The panel's state, which is the scene's: which mesh, where the case goes, and the settings that
# differ from their defaults. Only those, so that a default changed in `settings` reaches a scene
# saved before the change, rather than the scene holding on to the old one as if it had been chosen.
INPUT_MESH_REFERENCE = "InputMesh"
OUTPUT_DIRECTORY_PARAMETER = "OutputDirectory"
SETTINGS_PARAMETER = "Settings"

INVALID_COLOR = "#c83c28"

# The test run's settings are this computer's, not the scene's: where its svMultiPhysics is and how
# many cores it has are the same for every case on it, and wrong for the next computer the scene
# is opened on.
TEST_SETTINGS = "SimVascularMultiPhysics/Test/"
SOLVER_PATH_SETTING = TEST_SETTINGS + "SolverPath"
TEST_STEPS_SETTING = TEST_SETTINGS + "Steps"
TEST_PROCESSES_SETTING = TEST_SETTINGS + "Processes"
INSTALLERS_URL = "https://simtk.org/frs/?group_id=188"

# The test's result and its arrows are found by this attribute, naming the mesh they were run on,
# rather than referenced from the mesh: they are not saved with the scene -- a result is 40 MB, and
# the test is there to be run again -- and a reference would outlive them in the saved file.
TEST_OF_ATTRIBUTE = "SimVascularMultiPhysics.TestOf"
TEST_RESULT_ROLE = "SimVascularMultiPhysics.TestRole"
# What the mesh is faded to, as the ROM panel fades it for its results: the arrows are inside it.
TEST_MESH_OPACITY = 0.25
# Colour tables for the speed, in the order they are tried: Slicer's are not all in every build.
SPEED_COLOR_NODES = ("vtkMRMLColorTableNodeFileColdToHotRainbow.txt", "vtkMRMLColorTableNodeRainbow")
# How long a test asked to stop is given to finish its step and write it, before it is killed.
STOP_GRACE_MS = 120000


class SimVascularMultiPhysics(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SimVascular MultiPhysics")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SimVascular")]
        # Mesh Prep names the faces; the ROM panel shares the conditions and the table they are
        # set in, and its folder is where svromsetup is imported from.
        self.parent.dependencies = ["SimVascularMeshPrep", "SimVascularROM"]
        self.parent.contributors = ["Cardiovascular Biomechanics Computation Lab (Stanford University)"]
        self.parent.helpText = _(
            "Write an svMultiPhysics rigid-wall CFD case for a volume mesh whose faces are named "
            "in SimVascular Mesh Prep: the mesh-complete folder, the inflow waveforms and "
            "solver.xml. The boundary conditions are the ones SimVascular ROM Simulation sets "
            "for the same mesh; every other setting has a default and can be changed."
        )
        self.parent.acknowledgementText = _(
            "Developed in the Cardiovascular Biomechanics Computation Lab at Stanford "
            "University. The writing is the svmpsetup package beside this module, which runs "
            "outside Slicer as well."
        )


class SimVascularMultiPhysicsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._updating = False
        self.namedFaces = NamedMeshFaces()
        self.conditionsTable = None
        # The settings as they stand: what reads, by key, and what was typed and does not, with
        # why. Kept apart for the reason the conditions are -- a value that vanishes on a typo is
        # worse than one drawn in red -- and a setting in `invalid` blocks writing until it reads.
        self.values = settings.defaults()
        self.invalid = {}
        self.fields = {}
        self.labels = {}
        # Whether the panel has changed since the solver files were last written here: a test is
        # run on the files, so it writes them first when they would not say what the panel says,
        # and otherwise leaves them -- hand edits and all -- as they are.
        self._filesDirty = False
        # The test running, and what it was started with; None when there is none.
        self.process = None
        self.progress = None
        self._test = None
        self._partialLine = ""
        self._stopRequested = False
        self._solver = None
        self._solverLooked = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimVascularMultiPhysics.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimVascularMultiPhysicsLogic()
        self.conditionsTable = BoundaryConditionsTable(
            self.ui.facesTable, self.ui.valuesHintLabel, self.setStatus,
            changed=self.onConditionsChanged,
            startDirectory=lambda: self.ui.outputDirectoryPathLineEdit.currentPath)
        self.buildSettings()

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.openMeshPrepButton.connect("clicked(bool)", self.onOpenMeshPrep)
        self.ui.outputDirectoryPathLineEdit.connect("currentPathChanged(QString)",
                                                    self.onOutputDirectoryChanged)
        self.ui.resetSettingsButton.connect("clicked(bool)", self.onResetSettings)
        self.ui.createSolverFilesButton.connect("clicked(bool)", self.onCreateSolverFiles)

        self.restoreTestSettings()
        self.ui.testOutputTextEdit.setFont(qt.QFontDatabase.systemFont(qt.QFontDatabase.FixedFont))
        self.ui.testOutputTextEdit.setMaximumBlockCount(5000)
        self.ui.solverPathLineEdit.connect("currentPathChanged(QString)", self.onSolverPathChanged)
        self.ui.testStepsSpinBox.connect("valueChanged(int)", self.onTestSettingChanged)
        self.ui.testProcessesSpinBox.connect("valueChanged(int)", self.onTestSettingChanged)
        self.ui.runTestButton.connect("clicked(bool)", self.onRunTest)
        self.ui.showTestResultsButton.connect("clicked(bool)", self.onShowTestResults)
        # Looked for once the section is opened, not at startup: on Windows that asks WSL, which
        # takes seconds the first time it is woken.
        self.ui.testCollapsibleButton.connect("contentsCollapsed(bool)", self.onTestSectionCollapsed)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.restoreFromParameterNode()

    def enter(self):
        """Coming back, perhaps from Mesh Prep with the faces renamed, or from the ROM panel with
        the conditions changed: both are read again."""
        self.restoreFromParameterNode()

    def exit(self):
        clearHighlight()

    def cleanup(self):
        clearHighlight()
        self.removeObservers()
        # Slicer is closing: a test left running would outlive it, writing into a folder nobody
        # is reading.
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(5000)

    def onSceneEndImport(self, caller=None, event=None):
        self.restoreFromParameterNode()

    def onSceneEndClose(self, caller=None, event=None):
        self.onMeshChanged()

    # -- the settings, one field per row of svmpsetup.settings -----------------------------
    def buildSettings(self):
        """A collapsed group per section and a field per setting, from the table itself.

        Drawn from `settings.SETTINGS` rather than laid out in the .ui, so that a setting added
        there has a field here with nothing else to change, and a default changed there is the
        one shown.
        """
        groups = {}
        for section in settings.SECTIONS:
            group = ctk.ctkCollapsibleGroupBox()
            group.title = _(section)
            group.collapsed = True
            groups[section] = qt.QFormLayout(group)
            note = settings.SECTION_NOTES.get(section)
            if note:
                label = qt.QLabel(_(note))
                label.wordWrap = True
                groups[section].addRow(label)
            # The container's layout rather than the .ui's name for it: childWidgetVariables
            # finds widgets, and a layout is not one.
            self.ui.settingsContainer.layout().addWidget(group)
        for setting in settings.SETTINGS:
            label = qt.QLabel(_(setting.label) + ":")
            if setting.kind == settings.BOOL:
                field = qt.QCheckBox()
                field.connect("toggled(bool)", lambda _checked, key=setting.key: self.onSettingEdited(key))
            elif setting.choices:
                field = qt.QComboBox()
                for choice in setting.choices:
                    field.addItem(choice)
                field.connect("currentIndexChanged(int)",
                              lambda _index, key=setting.key: self.onSettingEdited(key))
            else:
                field = qt.QLineEdit()
                field.placeholderText = settings.BY_KEY[setting.key].format(setting.default)
                field.connect("editingFinished()", lambda key=setting.key: self.onSettingEdited(key))
            tip = setting.note + ("\n" if setting.note else "") + _("Default: {value}.").format(
                value=setting.format(setting.default))
            label.toolTip = tip
            field.toolTip = tip
            groups[setting.section].addRow(label, field)
            self.fields[setting.key] = field
            self.labels[setting.key] = label

    def fieldText(self, key):
        field = self.fields[key]
        if isinstance(field, qt.QCheckBox):
            return "true" if field.checked else "false"
        if isinstance(field, qt.QComboBox):
            return field.currentText
        return field.text

    def showSettings(self):
        """Every field from `self.values`, with the typed text kept where it did not read."""
        self._updating = True
        try:
            for setting in settings.SETTINGS:
                field = self.fields[setting.key]
                value = self.values[setting.key]
                if isinstance(field, qt.QCheckBox):
                    field.checked = bool(value)
                elif isinstance(field, qt.QComboBox):
                    field.currentIndex = max(0, field.findText(str(value)))
                else:
                    field.text = (self.invalid[setting.key][0] if setting.key in self.invalid
                                  else setting.format(value))
                self.styleSetting(setting.key)
        finally:
            self._updating = False

    def styleSetting(self, key):
        """Bold where the value is not the default, red where it does not read."""
        setting = settings.BY_KEY[key]
        label, field = self.labels[key], self.fields[key]
        font = label.font
        font.setBold(key not in self.invalid and self.values[key] != setting.default)
        label.font = font
        if key in self.invalid:
            field.styleSheet = f"color: {INVALID_COLOR};"
            field.toolTip = self.invalid[key][1]
        else:
            field.styleSheet = ""
            field.toolTip = label.toolTip

    def onSettingEdited(self, key):
        if self._updating:
            return
        text = self.fieldText(key)
        try:
            self.values[key] = settings.BY_KEY[key].parse(text)
            self.invalid.pop(key, None)
        except settings.SettingError as error:
            self.invalid[key] = (text, str(error))
        self._filesDirty = True
        self.styleSetting(key)
        self.saveToParameterNode()
        self.updateButtons()

    def onResetSettings(self):
        self.values = settings.defaults()
        self.invalid = {}
        self._filesDirty = True
        self.showSettings()
        self.saveToParameterNode()
        self.updateButtons()

    # -- state that is saved: the conditions on the mesh, the rest on the scene ---
    def restoreFromParameterNode(self):
        node = self.logic.getParameterNode()
        self._updating = True
        try:
            directory = node.GetParameter(OUTPUT_DIRECTORY_PARAMETER)
            if directory:
                self.ui.outputDirectoryPathLineEdit.currentPath = directory
            self.values = settings.defaults()
            self.invalid = {}
            try:
                saved = json.loads(node.GetParameter(SETTINGS_PARAMETER) or "{}")
            except ValueError:
                saved = {}
            for key, value in saved.items():
                if key not in settings.BY_KEY:
                    # A setting since removed; nothing to apply it to.
                    continue
                try:
                    self.values[key] = settings.BY_KEY[key].parse(value)
                except settings.SettingError:
                    logging.warning("SimVascular MultiPhysics could not read the saved %s.", key)
            mesh = node.GetNodeReference(INPUT_MESH_REFERENCE)
            if mesh is not None:
                wasBlocked = self.ui.inputMeshSelector.blockSignals(True)
                self.ui.inputMeshSelector.setCurrentNode(mesh)
                self.ui.inputMeshSelector.blockSignals(wasBlocked)
        finally:
            self._updating = False
        self.showSettings()
        self.onMeshChanged()
        # A scene opened is taken to have the files it was written with: a test runs on them as
        # they are, until something here is changed.
        self._filesDirty = False

    def saveToParameterNode(self):
        if self._updating:
            return
        node = self.logic.getParameterNode()
        wasModifying = node.StartModify()
        node.SetParameter(OUTPUT_DIRECTORY_PARAMETER, self.ui.outputDirectoryPathLineEdit.currentPath or "")
        changed = {key: value for key, value in self.values.items()
                   if value != settings.BY_KEY[key].default}
        node.SetParameter(SETTINGS_PARAMETER, json.dumps(changed, sort_keys=True))
        node.SetNodeReferenceID(INPUT_MESH_REFERENCE, self.ui.inputMeshSelector.currentNodeID or None)
        node.EndModify(wasModifying)

    def onOutputDirectoryChanged(self, *_args):
        if self._updating:
            return
        self._filesDirty = True
        self.saveToParameterNode()
        self.updateButtons()

    # -- the mesh -------------------------------------------------------------------
    def onMeshChanged(self, _node=None):
        """Read the selected mesh's names, measure its caps, and bring back its conditions."""
        self.ui.openMeshPrepButton.visible = False
        node = self.ui.inputMeshSelector.currentNode()
        self._filesDirty = True
        self.saveToParameterNode()
        if node is None or node.GetMesh() is None:
            self.ui.meshStatusLabel.text = ""
            self.conditionsTable.setMesh(None, None)
            self.updateButtons()
            return
        try:
            named = self.namedFaces.read(node)
        except ValueError as error:
            self.ui.meshStatusLabel.text = str(error)
            self.conditionsTable.setMesh(None, None)
            self.updateButtons()
            return
        text = self.namedFaces.describe()
        if not isinstance(node.GetMesh(), vtk.vtkUnstructuredGrid):
            text += " " + _("This is a surface: a 3D simulation is solved on the volume mesh CFD "
                            "Mesh Generator makes from it.")
        self.ui.meshStatusLabel.text = text
        self.ui.openMeshPrepButton.visible = not named or bool(self.namedFaces.unnamed)
        self.conditionsTable.setMesh(node if named else None, self.namedFaces if named else None)
        if named and not self.ui.outputDirectoryPathLineEdit.currentPath:
            self.ui.outputDirectoryPathLineEdit.currentPath = self.logic.suggestedOutputDirectory(node)
        self.updateButtons()

    def onConditionsChanged(self):
        self._filesDirty = True
        self.updateButtons()

    def onOpenMeshPrep(self):
        mesh = self.ui.inputMeshSelector.currentNode()
        slicer.util.selectModule("SimVascularMeshPrep")
        if mesh is not None:
            slicer.util.getModuleWidget("SimVascularMeshPrep").ui.inputMeshSelector.setCurrentNode(mesh)

    # -- writing the case ------------------------------------------------------------------
    def problems(self):
        """What stops the case being written, in words; empty when nothing does."""
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or not self.conditionsTable.caps:
            return [_("Select a volume mesh whose faces are named.")]
        found = []
        if not isinstance(node.GetMesh(), vtk.vtkUnstructuredGrid):
            found.append(_("A 3D simulation is solved on a volume mesh, and this is a surface."))
        found += case.problems(self.conditionsTable.conditionsByName(),
                               list(self.conditionsTable.capNames().values()), self.values)
        found += [reason for _text, reason in self.invalid.values()]
        if not self.ui.outputDirectoryPathLineEdit.currentPath:
            found.append(_("Choose an output folder."))
        return found

    def onCreateSolverFiles(self):
        with slicer.util.tryWithErrorDisplay(_("The solver files could not be created."),
                                             waitCursor=True):
            self.setStatus(self.createSolverFiles())

    def createSolverFiles(self):
        """Write the case into the output folder, and say what was written."""
        found = self.problems()
        if found:
            raise RuntimeError(" ".join(found))
        node = self.ui.inputMeshSelector.currentNode()
        directory = self.ui.outputDirectoryPathLineEdit.currentPath
        result = case.write_case(directory, node.GetMesh(), self.namedFaces.names,
                                 self.conditionsTable.conditionsByName(), self.values,
                                 face_id_array_name=self.namedFaces.arrayName)
        self._filesDirty = False
        return _("Wrote {path}: {faces} faces, {caps} of them caps, over {elements:,} elements"
                 "{waveforms}.").format(
            path=result.solver_xml, faces=result.faces, caps=result.caps,
            elements=result.elements,
            waveforms=(_(", and {count} inflow waveform(s)").format(count=len(result.flow_files))
                       if result.flow_files else ""))

    # -- panel state -----------------------------------------------------------------------
    def updateButtons(self):
        if self.conditionsTable is None:
            return
        found = self.problems()
        self.ui.createSolverFilesButton.enabled = not found
        self.ui.createSolverFilesButton.toolTip = (
            _("Write solver.xml, the mesh folder it reads and the inflow waveforms into the "
              "output folder.") if not found else " ".join(found))
        if self.conditionsTable.caps:
            self.setStatus(" ".join(found) if found else _("Ready to create the solver files."),
                           warning=bool(found))
        self.updateTestButtons(found)

    def setStatus(self, text, warning=False):
        self.ui.statusLabel.text = text
        self.ui.statusLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""

    # -- the test run: a few steps here, before the cluster ----------------------------------
    def restoreTestSettings(self):
        stored = qt.QSettings()
        self.ui.solverPathLineEdit.currentPath = str(stored.value(SOLVER_PATH_SETTING, "") or "")
        self.ui.testStepsSpinBox.value = int(stored.value(TEST_STEPS_SETTING, trial.DEFAULT_STEPS))
        self.ui.testProcessesSpinBox.value = int(stored.value(TEST_PROCESSES_SETTING,
                                                              trial.DEFAULT_PROCESSES))

    def onTestSettingChanged(self, *_args):
        stored = qt.QSettings()
        stored.setValue(TEST_STEPS_SETTING, self.ui.testStepsSpinBox.value)
        stored.setValue(TEST_PROCESSES_SETTING, self.ui.testProcessesSpinBox.value)

    def onSolverPathChanged(self, path):
        qt.QSettings().setValue(SOLVER_PATH_SETTING, path or "")
        self.solver(refresh=True)
        self.updateTestButtons()

    def onTestSectionCollapsed(self, collapsed):
        if not collapsed and not self._solverLooked:
            self.solver()
            self.updateTestButtons()

    def solver(self, refresh=False):
        """The svMultiPhysics a test runs, found with the environment it will run with."""
        if refresh or not self._solverLooked:
            self._solver = trial.find_solver(self.ui.solverPathLineEdit.currentPath or None,
                                             slicer.util.startupEnvironment())
            self._solverLooked = True
            self.ui.solverFoundLabel.text = (
                _("Found {solver}.").format(solver=self._solver.describe()) if self._solver
                else self.solverMissingText())
        return self._solver

    def solverMissingText(self):
        configured = self.ui.solverPathLineEdit.currentPath
        if configured:
            return _("There is no svmultiphysics executable at {path}.").format(path=configured)
        return _("svMultiPhysics was not found. Install it from {url}, with OpenMPI, LAPACK and "
                 "BLAS, or give the path of a build of your own.").format(url=INSTALLERS_URL)

    def updateTestButtons(self, found=None):
        if self.process is not None:
            self.ui.runTestButton.enabled = True    # it is the stop button while a test runs
            return
        reasons = list(self.problems() if found is None else found)
        if self._solverLooked and self._solver is None:
            reasons.append(self.solverMissingText())
        self.ui.runTestButton.enabled = not reasons
        self.ui.runTestButton.toolTip = (
            _("Write the solver files if they have changed, then run svMultiPhysics on them for "
              "the time steps below.") if not reasons else " ".join(reasons))

    def setTestReport(self, lines, error=False):
        self.ui.testReportLabel.text = "\n".join(lines)
        self.ui.testReportLabel.styleSheet = f"QLabel {{ color: {INVALID_COLOR}; }}" if error else ""

    def onRunTest(self):
        if self.process is not None:
            self.stopTest()
            return
        with slicer.util.tryWithErrorDisplay(_("The test could not be started."), waitCursor=True):
            self.startTest()

    def startTest(self):
        """Write the files if the panel has changed since they were, cut them to a few steps
        beside the case, and start svMultiPhysics on them; this returns while it runs."""
        found = self.problems()
        if found:
            raise RuntimeError(" ".join(found))
        solver = self.solver(refresh=True)
        if solver is None:
            raise RuntimeError(self.solverMissingText())
        directory = self.ui.outputDirectoryPathLineEdit.currentPath
        if self._filesDirty or not os.path.isfile(os.path.join(directory, case.SOLVER_XML_NAME)):
            self.setStatus(self.createSolverFiles())
        steps, processes = self.ui.testStepsSpinBox.value, self.ui.testProcessesSpinBox.value
        schedule = trial.schedule(directory)
        trialXml = trial.write_trial(directory, steps)
        argv = trial.command(solver, processes, trialXml.parent)
        ranks = processes if processes > 1 and solver.mpiexec else 1
        self.progress = trial.Progress(steps, schedule)
        header = _("Running {steps} step(s) of the case in {folder}, on {ranks} processor(s).").format(
            steps=steps, folder=trialXml.parent, ranks=ranks)
        if processes > 1 and not solver.mpiexec:
            header += " " + _("mpiexec was not found, so the test runs on one processor.")
        self._test = {"directory": str(trialXml.parent), "ranks": ranks, "schedule": schedule,
                      "started": time.time(), "header": header, "summary": []}
        self._partialLine = ""
        self._stopRequested = False

        process = qt.QProcess()
        process.setWorkingDirectory(str(trialXml.parent))
        environment = qt.QProcessEnvironment()
        for key, value in slicer.util.startupEnvironment().items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        # One stream: the solver's error is on stderr and its iterations on stdout, and read apart
        # the error would come out of order with the step it stopped at.
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        process.connect("readyReadStandardOutput()", self.onTestOutput)
        process.connect("finished(int,QProcess::ExitStatus)", self.onTestFinished)
        process.connect("errorOccurred(QProcess::ProcessError)", self.onTestError)
        self.process = process
        self.ui.testOutputTextEdit.clear()
        self.ui.testOutputTextEdit.appendPlainText("$ " + " ".join(argv))
        self.ui.showTestResultsButton.enabled = False
        self.ui.runTestButton.text = _("Stop test")
        self.setTestReport([header])
        process.start(argv[0], argv[1:])

    def readTestOutput(self):
        data = self.process.readAllStandardOutput()
        raw = data.data() if hasattr(data, "data") else data
        text = bytes(raw).decode("utf-8", errors="replace") if not isinstance(raw, str) else raw
        self._partialLine += text
        *lines, self._partialLine = self._partialLine.split("\n")
        for line in lines:
            self.takeTestLine(line)

    def takeTestLine(self, line):
        line = line.rstrip("\r")
        self.ui.testOutputTextEdit.appendPlainText(line)
        self.progress.feed(line)

    def onTestOutput(self):
        if self.process is None:
            return
        self.readTestOutput()
        current = self.progress.current
        if current is None:
            progress = _("Setting up: {time} so far.").format(
                time=trial.duration(time.time() - self._test["started"]))
        else:
            progress = _("Step {step} of {steps}, iteration {iteration}.").format(
                step=current.step, steps=self.progress.steps, iteration=current.iteration)
            perStep = self.progress.seconds_per_step()
            if perStep and len(self.progress.last) > 1:
                progress += " " + _("{time} per step so far.").format(time=trial.duration(perStep))
        self.setTestReport([self._test["header"], progress] + self.progress.errors,
                           error=bool(self.progress.errors))
        # svMultiPhysics does not stop for a NaN; this does.
        if self.progress.nan_step is not None and not self._stopRequested:
            self.stopTest()

    def stopTest(self):
        """Ask the test to stop after the step it is on, as svMultiPhysics is asked: by its stop
        file. Asked a second time, or not stopped in time, it is killed."""
        if self.process is None:
            return
        if self._stopRequested:
            self.process.kill()
            return
        self._stopRequested = True
        try:
            trial.stop(self._test["directory"])
        except Exception:
            logging.warning("The test's stop file could not be written; killing it.", exc_info=True)
            self.process.kill()
            return
        self.ui.runTestButton.text = _("Stopping after this step (press again to kill)")
        started = self._test["started"]
        qt.QTimer.singleShot(STOP_GRACE_MS, lambda: self.killTest(started))

    def killTest(self, started):
        if self.process is not None and self._test and self._test["started"] == started:
            self.process.kill()

    def onTestFinished(self, exitCode, exitStatus):
        if self.process is None:
            return
        self.readTestOutput()
        if self._partialLine:
            self.takeTestLine(self._partialLine)
            self._partialLine = ""
        report = self.progress.finish(exitCode, crashed=exitStatus == qt.QProcess.CrashExit,
                                      stopped=self._stopRequested)
        results = trial.results_folder(self._test["directory"], self._test["ranks"])
        files = trial.result_files(results)
        lines = trial.summary(report, self._test["schedule"], self._test["ranks"],
                              files[-1].stat().st_size if files else None)
        self._test["results"] = str(results)
        self._test["summary"] = lines
        self._test["report"] = report
        self.setTestReport(lines, error=bool(report.errors))
        self.process = None
        self.ui.runTestButton.text = _("Run test")
        self.ui.showTestResultsButton.enabled = bool(files)
        self.updateTestButtons()

    def onTestError(self, error):
        """Only a test that never started is dealt with here: any other ends in `finished`."""
        if self.process is None or error != qt.QProcess.FailedToStart:
            return
        self.setTestReport([_("svMultiPhysics could not be started: {reason}.").format(
            reason=self.process.errorString())], error=True)
        self.process = None
        self.ui.runTestButton.text = _("Run test")
        self.updateTestButtons()

    def onShowTestResults(self):
        with slicer.util.tryWithErrorDisplay(_("The test's results could not be shown."),
                                             waitCursor=True):
            self.showTestResults()

    def testNode(self, mesh, role, name):
        """The test's `role` node for `mesh`, made the first time; never saved with the scene."""
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if node.GetAttribute(TEST_OF_ATTRIBUTE) == mesh.GetID() and \
                    node.GetAttribute(TEST_RESULT_ROLE) == role:
                break
        else:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            node.SetAttribute(TEST_OF_ATTRIBUTE, mesh.GetID())
            node.SetAttribute(TEST_RESULT_ROLE, role)
            node.SetSaveWithScene(False)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSaveWithScene(False)
        node.SetName(name)
        node.SetAndObserveTransformNodeID(mesh.GetTransformNodeID())
        return node

    def showTestResults(self):
        """The test's last step: arrows along the velocity, coloured by speed, inside the faded
        mesh; the result itself beside them, hidden, for the Models module; and which way the
        flow went through each face, against which way its condition says it should."""
        mesh = self.ui.inputMeshSelector.currentNode()
        files = trial.result_files(self._test["results"]) if self._test and self._test.get("results") else []
        if mesh is None or not files:
            raise RuntimeError(_("There is no test result to show: run the test first."))
        last = files[-1]
        step = int(last.stem.rsplit("_", 1)[-1])
        grid = trial.read_result(last)

        result = self.testNode(mesh, "result", _("{mesh} test result").format(mesh=mesh.GetName()))
        result.SetAndObserveMesh(grid)
        result.GetDisplayNode().SetVisibility(False)

        arrows = self.testNode(mesh, "velocity", _("{mesh} test velocity").format(mesh=mesh.GetName()))
        arrows.SetAndObserveMesh(trial.velocity_arrows(grid))
        display = arrows.GetDisplayNode()
        display.SetActiveScalarName(trial.SPEED_ARRAY)
        display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        for colorNodeId in SPEED_COLOR_NODES:
            if slicer.mrmlScene.GetNodeByID(colorNodeId) is not None:
                display.SetAndObserveColorNodeID(colorNodeId)
                break
        display.SetScalarRangeFlag(display.UseDataScalarRange)
        display.SetScalarVisibility(True)
        display.SetVisibility(True)
        try:
            colors = slicer.modules.colors.logic()
            legend = colors.GetColorLegendDisplayNode(arrows) or colors.AddDefaultColorLegendDisplayNode(arrows)
            legend.SetTitleText(_("Speed (cm/s)"))
            legend.SetLabelFormat("%.1f")
            legend.SetVisibility(True)
            legend.SetSaveWithScene(False)
        except Exception:
            logging.debug("SimVascular MultiPhysics could not show a colour legend.", exc_info=True)
        meshDisplay = mesh.GetDisplayNode()
        if meshDisplay is not None and meshDisplay.GetOpacity() > TEST_MESH_OPACITY:
            meshDisplay.SetOpacity(TEST_MESH_OPACITY)
        # What else is drawn inside the same mesh -- the 0D panel's centerlines and coloured tubes,
        # which the mesh references -- is hidden, legend and all: solid inside the faded mesh, they
        # hide the arrows, and a second colour scale beside the speed's reads as its.
        for index in range(mesh.GetNumberOfNodeReferenceRoles()):
            other = mesh.GetNodeReference(mesh.GetNthNodeReferenceRole(index))
            if other is not None and other is not mesh and other.IsA("vtkMRMLModelNode"):
                for displayIndex in range(other.GetNumberOfDisplayNodes()):
                    other.GetNthDisplayNode(displayIndex).SetVisibility(False)

        lines = list(self._test.get("summary") or [])
        fluxFile = os.path.join(self._test["results"], trial.FLUX_FILE)
        if os.path.isfile(fluxFile):
            integrals = trial.read_face_integrals(fluxFile)
            row = integrals.at(step)
            flows = trial.face_flows(integrals, self.conditionsTable.conditionsByName(), step)
            lines += [""] + trial.flow_lines(flows, row[1] if row else None)
            wrong = any(not flow.right_way for flow in flows)
        else:
            lines += ["", trial.flow_lines([])[0]]
            wrong = False
        report = self._test.get("report")
        self.setTestReport(lines, error=wrong or bool(report and report.errors))
        return arrows


class SimVascularMultiPhysicsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def suggestedOutputDirectory(meshNode):
        """`svmultiphysics/<mesh>` beside the saved scene, once it has been saved; nowhere until."""
        url = slicer.mrmlScene.GetURL()
        directory = os.path.dirname(url) if url else ""
        if not os.path.isdir(directory):
            return ""
        return os.path.join(directory, "svmultiphysics", sanitized(meshNode.GetName()) or "model")


class SimVascularMultiPhysicsTest(ScriptedLoadableModuleTest):
    """Runs under Slicer; the package's own tests run without it."""

    def setUp(self):
        slicer.mrmlScene.Clear()
        slicer.mrmlScene.SetURL("")

    def runTest(self):
        self.setUp()
        self.test_theConditionsAreTheROMPanelsOwn()
        self.setUp()
        self.test_aSettingIsKeptAndATypoRefused()
        self.setUp()
        self.test_theSolverFilesAreWritten()
        self.setUp()
        self.test_aSolverAskedForAndNotThereIsSaidSo()
        self.setUp()
        self.test_aTestRunStepsTheCaseAndSaysWhichWayItFlows()

    def namedCube(self, name="cube"):
        from svmeshcomplete import testing

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObserveMesh(testing.cube_mesh(4))
        node.CreateDefaultDisplayNodes()
        node.SetAttribute(FACE_NAMES_ATTRIBUTE, json.dumps(
            {str(testing.WALL_ID): "wall", str(testing.INLET_ID): "cap_inlet",
             str(testing.OUTLET_ID): "cap_outlet"}))
        return node

    def widget(self):
        widget = slicer.util.getModuleWidget("SimVascularMultiPhysics")
        # The widget outlives the scene, so what an earlier test left on it is reset here.
        widget.ui.outputDirectoryPathLineEdit.currentPath = ""
        widget.ui.solverPathLineEdit.currentPath = ""
        widget.onResetSettings()
        return widget

    @staticmethod
    def typeCondition(table, faceId, kind, text):
        row = table.rowOf(faceId)
        table.table.cellWidget(row, TYPE_COLUMN).currentIndex = KIND_CHOICES.index(kind)
        table.table.item(row, VALUES_COLUMN).setText(text)

    def test_theConditionsAreTheROMPanelsOwn(self):
        """Set in either panel, seen in the other: one set per mesh."""
        from svmeshcomplete import testing

        node = self.namedCube()
        rom = slicer.util.getModuleWidget("SimVascularROM")
        rom.ui.inputMeshSelector.setCurrentNode(node)
        self.typeCondition(rom.conditionsTable, testing.OUTLET_ID, bcs.RCR_KIND, "100, 1e-4, 1000")

        widget = self.widget()
        widget.ui.inputMeshSelector.setCurrentNode(node)
        self.assertEqual(widget.conditionsTable.conditions,
                         {testing.OUTLET_ID: bcs.RCR(100.0, 1e-4, 1000.0, 0.0)})
        self.typeCondition(widget.conditionsTable, testing.INLET_ID, bcs.INFLOW, "12")
        rom.enter()
        self.assertEqual(rom.conditionsTable.conditions[testing.INLET_ID], bcs.Inflow.steady(12.0))
        self.delayDisplay("The conditions are shared with the ROM panel")

    def test_aSettingIsKeptAndATypoRefused(self):
        """A changed setting is saved with the scene; one svMultiPhysics would misread is refused."""
        from svmeshcomplete import testing

        widget = self.widget()
        node = self.namedCube()
        widget.ui.inputMeshSelector.setCurrentNode(node)
        widget.ui.outputDirectoryPathLineEdit.currentPath = slicer.app.temporaryPath
        self.typeCondition(widget.conditionsTable, testing.INLET_ID, bcs.INFLOW, "10")
        self.typeCondition(widget.conditionsTable, testing.OUTLET_ID, bcs.RESISTANCE_KIND, "1000")
        self.assertTrue(widget.ui.createSolverFilesButton.enabled)

        field = widget.fields["general.time_step_size"]
        field.text = "1,5e-4"
        widget.onSettingEdited("general.time_step_size")
        self.assertIn("general.time_step_size", widget.invalid)
        self.assertFalse(widget.ui.createSolverFilesButton.enabled)

        field.text = "1e-4"
        widget.onSettingEdited("general.time_step_size")
        self.assertTrue(widget.ui.createSolverFilesButton.enabled)
        self.assertTrue(widget.labels["general.time_step_size"].font.bold(), "a changed value is bold")
        saved = json.loads(widget.logic.getParameterNode().GetParameter(SETTINGS_PARAMETER))
        self.assertEqual(saved, {"general.time_step_size": 1e-4}, "only what differs is saved")
        self.delayDisplay("Settings kept, typos refused")

    def test_theSolverFilesAreWritten(self):
        import tempfile

        from svmeshcomplete import testing

        widget = self.widget()
        node = self.namedCube()
        widget.ui.inputMeshSelector.setCurrentNode(node)
        directory = tempfile.mkdtemp()
        widget.ui.outputDirectoryPathLineEdit.currentPath = directory
        self.typeCondition(widget.conditionsTable, testing.INLET_ID, bcs.INFLOW, "10")
        self.typeCondition(widget.conditionsTable, testing.OUTLET_ID, bcs.RCR_KIND, "121, 1.5e-4, 1212")
        said = widget.createSolverFiles()
        self.assertIn("3 faces", said)
        root = case.check_case(os.path.join(directory, case.SOLVER_XML_NAME))
        self.assertEqual(root.findtext("Add_mesh/Mesh_scale_factor"), "0.1")
        self.delayDisplay("Solver files written")

    def readyCube(self, widget):
        """The cube with both its caps given conditions, written into a folder of its own."""
        import tempfile

        from svmeshcomplete import testing

        node = self.namedCube()
        widget.ui.inputMeshSelector.setCurrentNode(node)
        directory = os.path.join(tempfile.mkdtemp(), "cube")
        widget.ui.outputDirectoryPathLineEdit.currentPath = directory
        self.typeCondition(widget.conditionsTable, testing.INLET_ID, bcs.INFLOW, "10")
        self.typeCondition(widget.conditionsTable, testing.OUTLET_ID, bcs.RCR_KIND, "121, 1.5e-4, 1212")
        return directory

    def test_aSolverAskedForAndNotThereIsSaidSo(self):
        """Not quietly replaced by another build: a test passed by the wrong one tests nothing."""
        widget = self.widget()
        self.readyCube(widget)
        widget.ui.solverPathLineEdit.currentPath = "/nowhere/svmultiphysics"
        self.assertFalse(widget.ui.runTestButton.enabled)
        self.assertIn("There is no svmultiphysics executable at /nowhere/svmultiphysics",
                      widget.ui.solverFoundLabel.text)
        self.delayDisplay("A solver that is not there is said not to be")

    def test_aTestRunStepsTheCaseAndSaysWhichWayItFlows(self):
        """Two steps of the cube, run as the panel runs them; skipped without svMultiPhysics."""
        widget = self.widget()
        if widget.solver(refresh=True) is None:
            self.delayDisplay("Skipped: svMultiPhysics is not installed here")
            return
        directory = self.readyCube(widget)
        widget.ui.testStepsSpinBox.value = 2
        widget.ui.testProcessesSpinBox.value = 1
        self.assertTrue(widget.ui.runTestButton.enabled, widget.ui.runTestButton.toolTip)
        widget.onRunTest()
        self.assertTrue(os.path.isfile(os.path.join(directory, case.SOLVER_XML_NAME)),
                        "the files are written before they are tested")
        deadline = time.time() + 300
        while widget.process is not None and time.time() < deadline:
            slicer.app.processEvents()
            time.sleep(0.05)
        report = widget.ui.testReportLabel.text
        self.assertIn("2 of 2 steps", report)
        self.assertIn("No errors.", report)
        self.assertTrue(os.path.isfile(os.path.join(directory + trial.TRIAL_SUFFIX, "1-procs",
                                                    "result_002.vtu")))
        self.assertTrue(widget.ui.showTestResultsButton.enabled)

        arrows = widget.showTestResults()
        self.assertGreater(arrows.GetMesh().GetNumberOfPoints(), 0)
        self.assertFalse(arrows.GetSaveWithScene(), "a test result is not saved with the scene")
        self.assertIn("comes in through every inflow", widget.ui.testReportLabel.text)
        self.delayDisplay("Two steps tested, and the flow goes the right way")
