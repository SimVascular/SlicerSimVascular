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
# Reload, in developer mode, runs this file again but not the package it imports, which Python keeps
# as it was first imported: a panel newer than its package then calls functions that do not take
# what it passes -- "estimate() got an unexpected keyword argument" -- in every callback, and Reload
# and Test stops part way, leaving the panel half set. So the package is imported afresh with the
# file. Only this module's own: svromsetup's classes are shared with the ROM panel's table, and an
# Inflow made by one copy is not an Inflow to the other.
for _name in [name for name in sys.modules if name == "svmpsetup" or name.startswith("svmpsetup.")]:
    del sys.modules[_name]

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
from svmpsetup import case, settings, time_step, trial
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
# The CFL number the time step is estimated at: the case's, with the settings, but not one of them --
# nothing in solver.xml is it.
CFL_PARAMETER = "CFLNumber"
# How the run's length is set -- in cardiac cycles, or as a number of time steps -- and how many
# cycles. Not settings either: in cycles, Number_of_time_steps is what they come to, and that is
# what is written.
RUN_LENGTH_PARAMETER = "RunLength"
CYCLES_PARAMETER = "CardiacCycles"
RUN_BY_CYCLES, RUN_BY_STEPS = "cycles", "steps"
# How the time step is set: user-defined, or estimated from the CFL number.
TIME_STEP_MODE_PARAMETER = "TimeStepMode"
# A max velocity (cm/s) and a mesh size (mm) typed over the case's, or empty when left to it.
CFL_VELOCITY_PARAMETER = "CFLMaxVelocity"
CFL_MESH_SIZE_PARAMETER = "CFLMeshSize"
# Whether the estimate adapts the saving to the step it sets: "false" once a saving setting is typed,
# so that a scene opened again keeps the saving as typed rather than adapting it once more.
CFL_ADAPT_SAVING_PARAMETER = "CFLAdaptSaving"
STEP_BY_USER, STEP_BY_CFL = "user", "cfl"
STEPS_KEY = "general.number_of_time_steps"
TIME_STEP_KEY = "general.time_step_size"
SAVING_KEYS = ("general.increment_in_saving_vtk_files", "general.start_saving_after_time_step",
               "general.increment_in_saving_restart_files")

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
            self.fields[setting.key] = field
            # The run's length and its time step each a section of their own, with a choice of how
            # it is set; every other setting a row.
            if setting.key == STEPS_KEY:
                label = self.addStepsSection(groups[setting.section], field, tip)
            elif setting.key == TIME_STEP_KEY:
                label = self.addTimeStepSection(groups[setting.section], field, tip)
            else:
                groups[setting.section].addRow(label, field)
            self.labels[setting.key] = label

    def addStepsSection(self, form, field, tip):
        """Number of time steps: user-defined, or calculated from cardiac cycles.

        One or the other, never both: two fields that each set the run's length would disagree the
        moment either was changed. User-defined by default (the user's choice, 2026-10-15, over the
        cycles it first defaulted to): the steps are the number typed, 10000, as the case was always
        written. The cycles are there because a run is read by the cycle -- the last one, or the
        last few, once the outlets have filled -- and a number of steps is a number of cycles only
        by arithmetic that has to be redone whenever the time step changes.
        Calculated, the steps are shown, greyed, and are what solver.xml gets; user-defined, they
        are typed and the cycles are not used. A steady inflow has no cycle to count, so with one
        the steps are typed whichever is chosen -- greyed, they would be a number nobody could set.
        Returns the label a changed value is bolded on.
        """
        box = qt.QGroupBox(_("Number of time steps"))
        grid = qt.QGridLayout(box)
        self.runLengthGroup = qt.QButtonGroup(box)
        self.cyclesRadio = qt.QRadioButton(_("Calculate from cardiac cycles:"))
        self.stepsRadio = qt.QRadioButton(_("User-defined:"))
        self.stepsRadio.checked = True
        for button in (self.cyclesRadio, self.stepsRadio):
            self.runLengthGroup.addButton(button)
        self.cyclesSpinBox = qt.QSpinBox()
        self.cyclesSpinBox.minimum = 1
        self.cyclesSpinBox.maximum = 1000
        self.cyclesSpinBox.value = time_step.DEFAULT_CYCLES
        self.cyclesSpinBox.suffix = _(" cycles")
        self.cyclesTip = _("The number of time steps, worked out from this many cardiac cycles of the "
                           "inflow waveform and the time step. Changing the number chooses it. "
                           "{cycles} by default.").format(
            cycles=time_step.DEFAULT_CYCLES)
        self.cyclesRadio.toolTip = self.cyclesSpinBox.toolTip = self.cyclesTip
        self.stepsRadio.toolTip = tip
        # Why the cycles calculate nothing, when they are chosen and cannot: said, under them, rather
        # than the box greyed -- greyed, it read as broken.
        self.cyclesNoteLabel = qt.QLabel()
        self.cyclesNoteLabel.wordWrap = True
        self.cyclesNoteLabel.styleSheet = "QLabel { color: #d08000; }"
        self.cyclesNoteLabel.visible = False
        # User-defined first, in both sections, so that they read alike, and chosen by default in both.
        grid.addWidget(self.stepsRadio, 0, 0)
        grid.addWidget(field, 0, 1)
        grid.addWidget(self.cyclesRadio, 1, 0)
        grid.addWidget(self.cyclesSpinBox, 1, 1)
        grid.addWidget(self.cyclesNoteLabel, 2, 0, 1, 2)
        form.addRow(box)
        self.cyclesRadio.connect("toggled(bool)", lambda _checked: self.onTimeSteppingChanged())
        self.cyclesSpinBox.connect("valueChanged(int)", lambda _value: self.onCyclesTyped())
        return self.stepsRadio

    def onCyclesTyped(self):
        """The cycles changed: the run calculated from them, which is what changing them asks for.
        The box is never greyed for it -- with User-defined chosen, a greyed box the user could not
        type in read as broken."""
        if self._updating:
            return
        self.setChecked(self.cyclesRadio, True)
        self.onTimeSteppingChanged()

    def addTimeStepSection(self, form, field, tip):
        """Time step size: user-defined, or estimated from a CFL number.

        Estimated, the step is the CFL number times a mesh size over a max velocity, both shown
        under it and filled in from the case when the estimate is chosen: the mesh's mean edge
        length, and the fastest velocity an inflow imposes. Either can be typed over -- a narrowing
        faster than the inflow, a finer region than the mean -- and is then bold, and kept as the
        case changes; one not typed over follows the mesh and the conditions. Use the case's values
        puts both back. The step is worked out again whenever any of it changes, shown greyed, with
        how it was found; the saving is kept as often in time as it was, unless a saving setting is
        typed, which clears Adapt the saving and leaves the saving as typed. The estimate is
        `svmpsetup.time_step`'s. None of this is a row of `settings`, which is what the writer puts
        in solver.xml: it is a way of choosing the step, and the step is what is written. Returns
        the label a changed value is bolded on.
        """
        box = qt.QGroupBox(_("Time step size"))
        grid = qt.QGridLayout(box)
        self.timeStepGroup = qt.QButtonGroup(box)
        self.userTimeStepRadio = qt.QRadioButton(_("User-defined:"))
        self.userTimeStepRadio.checked = True
        self.cflRadio = qt.QRadioButton(_("Estimate time step from CFL"))
        for button in (self.userTimeStepRadio, self.cflRadio):
            self.timeStepGroup.addButton(button)
        self.userTimeStepRadio.toolTip = tip
        self.cflRadio.toolTip = _(
            "The time step that is a CFL number: the mesh size over the max velocity, with the mesh's "
            "millimetres scaled to the solver's centimetres, rounded down to whole steps per cardiac "
            "cycle, and the saving kept as often in time as it was unless it is set by hand.")
        grid.addWidget(self.userTimeStepRadio, 0, 0)
        grid.addWidget(field, 0, 1)
        grid.addWidget(self.cflRadio, 1, 0, 1, 2)

        # What the estimate is made from, under it and indented, enabled only when it is chosen.
        self.cflInputs = qt.QWidget()
        inputs = qt.QFormLayout(self.cflInputs)
        inputs.setContentsMargins(24, 0, 0, 0)
        self.cflSpinBox = qt.QDoubleSpinBox()
        self.cflSpinBox.decimals = 2
        self.cflSpinBox.minimum = 0.01
        self.cflSpinBox.maximum = 100.0
        self.cflSpinBox.singleStep = 0.1
        self.cflSpinBox.value = time_step.DEFAULT_CFL
        self.cflSpinBox.toolTip = _("The Courant number: how many elements the max velocity crosses in "
                                    "one step. 1 is recommended for svMultiPhysics.")
        self.velocitySpinBox = qt.QDoubleSpinBox()
        self.velocitySpinBox.decimals = 2
        self.velocitySpinBox.minimum = 0.0
        self.velocitySpinBox.maximum = 1e6
        self.velocitySpinBox.suffix = " cm/s"
        self.velocitySpinBox.toolTip = _(
            "The fastest velocity in the model, in cm/s. Filled in with the fastest an inflow imposes: "
            "its peak flow, in mL/s, over its cap's area, in mm², twice that on the centreline with the "
            "Parabolic profile. Type over it for a faster flow the inflow does not show, through a "
            "narrowing for one.")
        self.meshSizeSpinBox = qt.QDoubleSpinBox()
        self.meshSizeSpinBox.decimals = 4
        self.meshSizeSpinBox.minimum = 0.0
        self.meshSizeSpinBox.maximum = 1e4
        self.meshSizeSpinBox.suffix = " mm"
        self.meshSizeSpinBox.toolTip = _(
            "The element size the step is for, in mm, as the mesh's coordinates are. Filled in with the "
            "mesh's mean edge length. Type over it for a region finer than the mean.")
        self.cflLabel = qt.QLabel(_("CFL number:"))
        self.velocityLabel = qt.QLabel(_("Max velocity:"))
        self.meshSizeLabel = qt.QLabel(_("Mesh size:"))
        for label, spinBox in ((self.cflLabel, self.cflSpinBox), (self.velocityLabel, self.velocitySpinBox),
                               (self.meshSizeLabel, self.meshSizeSpinBox)):
            label.toolTip = spinBox.toolTip
            inputs.addRow(label, spinBox)
        self.caseValuesButton = qt.QPushButton(_("Use the case's values"))
        # Always clickable: disabled until something was typed over, it read as broken.
        self.caseValuesButton.toolTip = _("Fill in the max velocity and the mesh size from the case: "
                                          "the fastest inflow and the mesh's mean edge length. A value "
                                          "typed over is replaced.")
        inputs.addRow(self.caseValuesButton)
        # The saving the estimate adapts, or leaves as typed: a box rather than each field bolded
        # when typed over, so that which it is shows, and the adapting can be asked for back.
        self.adaptSavingCheckBox = qt.QCheckBox(_("Adapt the saving to the time step"))
        self.adaptSavingCheckBox.checked = True
        self.adaptSavingCheckBox.toolTip = _(
            "Save results and restart files as often in time as before, whatever the step, with the "
            "results at the same points of every cardiac cycle. Type a saving setting under General "
            "and it is kept as typed instead: this is cleared, and the step only rounded to whole "
            "steps per cycle.")
        inputs.addRow(self.adaptSavingCheckBox)
        self.timeStepEstimateLabel = qt.QLabel()
        self.timeStepEstimateLabel.wordWrap = True
        self.timeStepEstimateLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        inputs.addRow(self.timeStepEstimateLabel)
        grid.addWidget(self.cflInputs, 2, 0, 1, 2)
        form.addRow(box)

        # Which of the two were typed over, and so are kept as the case changes.
        self._velocityByHand = False
        self._meshSizeByHand = False
        self.cflRadio.connect("toggled(bool)", self.onTimeStepModeChanged)
        self.cflSpinBox.connect("valueChanged(double)", lambda _value: self.onTimeSteppingChanged())
        self.velocitySpinBox.connect("valueChanged(double)", lambda _value: self.onCFLInputTyped("velocity"))
        self.meshSizeSpinBox.connect("valueChanged(double)", lambda _value: self.onCFLInputTyped("meshSize"))
        self.caseValuesButton.connect("clicked(bool)", lambda _checked: self.useCaseValues())
        self.adaptSavingCheckBox.connect("toggled(bool)", lambda _checked: self.onTimeSteppingChanged())
        self._edgeLength = None   # (mesh node id, mesh modified time, mean edge length)
        return self.userTimeStepRadio

    def onTimeStepModeChanged(self, checked):
        """The estimate chosen: filled in afresh from the case, whatever was typed before, and the
        saving adapted to it."""
        if checked and not self._updating:
            self._velocityByHand = self._meshSizeByHand = False
            self.setChecked(self.adaptSavingCheckBox, True)
        self.onTimeSteppingChanged()

    def onCFLInputTyped(self, which):
        if self._updating:
            return
        if which == "velocity":
            self._velocityByHand = True
        else:
            self._meshSizeByHand = True
        self.onTimeSteppingChanged()

    def useCaseValues(self):
        self._velocityByHand = self._meshSizeByHand = False
        self.onTimeSteppingChanged()

    def setChecked(self, button, checked):
        wasUpdating, self._updating = self._updating, True
        try:
            button.checked = checked
        finally:
            self._updating = wasUpdating

    def setSpinBox(self, spinBox, value):
        wasUpdating, self._updating = self._updating, True
        try:
            spinBox.value = value
        finally:
            self._updating = wasUpdating

    def runByCycles(self):
        return bool(self.cyclesRadio.checked)

    def cyclesToRun(self):
        """The cycles the steps are calculated from, or None when they are user-defined."""
        return self.cyclesSpinBox.value if self.runByCycles() else None

    def timeStepByCFL(self):
        return bool(self.cflRadio.checked)

    def hasWaveform(self):
        return bool(self.conditionsTable) and time_step.has_waveform(self.conditionsTable.conditionsByName())

    def adaptsSaving(self):
        return self.timeStepByCFL() and bool(self.adaptSavingCheckBox.checked)

    def derived(self, key):
        """Whether a setting is worked out here, not typed, and so not bolded. The steps and the
        step are greyed as well; the saving is not, since typing over it is how it is set by hand."""
        if key == STEPS_KEY:
            return self.runByCycles() and self.hasWaveform()
        if key == TIME_STEP_KEY:
            return self.timeStepByCFL()
        return key in SAVING_KEYS and self.adaptsSaving()

    def onTimeSteppingChanged(self):
        if self._updating:
            return
        self._filesDirty = True
        self.updateDerived()
        self.saveToParameterNode()
        self.updateButtons()

    def setDerivedValue(self, key, value):
        """A worked-out value into its field, as if it had been read from it."""
        self.values[key] = value
        self.invalid.pop(key, None)
        wasUpdating, self._updating = self._updating, True
        try:
            self.fields[key].text = settings.BY_KEY[key].format(value)
        finally:
            self._updating = wasUpdating
        self.styleSetting(key)

    def updateDerived(self):
        """The time step from the CFL, then the steps from the cycles, where those are chosen; and
        which fields can be typed in. In that order: the steps follow the step."""
        byCFL, byCycles = self.timeStepByCFL(), self.runByCycles()
        # Greyed as well as read-only: which of the two is being set should show at a glance.
        for key in (TIME_STEP_KEY, STEPS_KEY):
            derived = self.derived(key)
            self.fields[key].readOnly = derived
            self.fields[key].enabled = not derived
        self.cflInputs.enabled = byCFL
        self.cflInputs.visible = byCFL
        if byCFL:
            self.updateTimeStepFromCFL()
        self.updateStepsFromCycles()
        for key in (TIME_STEP_KEY, STEPS_KEY) + SAVING_KEYS:
            self.styleSetting(key)

    def updateStepsFromCycles(self):
        """Number of time steps from the cardiac cycles, when it is calculated from them."""
        self.cyclesNoteLabel.visible = False
        if not self.runByCycles():
            return
        conditions = self.conditionsTable.conditionsByName() if self.conditionsTable else {}
        step = self.values.get(TIME_STEP_KEY)
        steps = (time_step.steps_for_cycles(self.cyclesSpinBox.value, conditions, step)
                 if TIME_STEP_KEY not in self.invalid else None)
        if steps is None:
            # No cycle to count, or no step to count it in: the steps as they stand, and said why.
            if TIME_STEP_KEY in self.invalid:
                note = _("The time step does not read, so no steps can be calculated from the cycles.")
            elif not self.conditionsTable or not self.conditionsTable.caps:
                note = _("Select a mesh with an inflow waveform to calculate the steps from its cycles.")
            else:
                note = _("The inflow is steady, with no cardiac cycle to count: the run is the number "
                         "of time steps typed above.")
            self.cyclesNoteLabel.text = note
            self.cyclesNoteLabel.visible = True
            return
        self.setDerivedValue(STEPS_KEY, steps)

    def showEstimate(self, text, warning=False):
        self.timeStepEstimateLabel.text = text
        self.timeStepEstimateLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""

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
        font.setBold(key not in self.invalid and self.values[key] != setting.default
                     and not self.derived(key))
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
        before = self.values.get(key)
        try:
            self.values[key] = settings.BY_KEY[key].parse(text)
            self.invalid.pop(key, None)
        except settings.SettingError as error:
            self.invalid[key] = (text, str(error))
        self._filesDirty = True
        # A saving setting typed while the estimate adapts the saving: the user's from now on, kept
        # as typed. Only one changed -- a field left without a change finishes its editing too.
        if (key in SAVING_KEYS and self.adaptsSaving()
                and (key in self.invalid or self.values[key] != before)):
            self.setChecked(self.adaptSavingCheckBox, False)
        self.styleSetting(key)
        if key in (TIME_STEP_KEY, "bc.inflow_profile", "bc.impose_flux", "mesh.scale_factor") + SAVING_KEYS:
            self.updateDerived()
        self.saveToParameterNode()
        self.updateButtons()

    def onResetSettings(self):
        self.values = settings.defaults()
        self.invalid = {}
        self._filesDirty = True
        wasUpdating, self._updating = self._updating, True
        try:
            self.stepsRadio.checked = True
            self.cyclesSpinBox.value = time_step.DEFAULT_CYCLES
            self.userTimeStepRadio.checked = True
            self.cflSpinBox.value = time_step.DEFAULT_CFL
            self._velocityByHand = self._meshSizeByHand = False
            self.adaptSavingCheckBox.checked = True
        finally:
            self._updating = wasUpdating
        self.showSettings()
        self.updateDerived()
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
            try:
                self.cflSpinBox.value = float(node.GetParameter(CFL_PARAMETER) or time_step.DEFAULT_CFL)
            except ValueError:
                self.cflSpinBox.value = time_step.DEFAULT_CFL
            # Set in cycles only when it was saved so; a scene from before the run could be set in
            # cycles has its steps user-defined, as they were written.
            runLength = node.GetParameter(RUN_LENGTH_PARAMETER)
            (self.cyclesRadio if runLength == RUN_BY_CYCLES else self.stepsRadio).checked = True
            byCFL = node.GetParameter(TIME_STEP_MODE_PARAMETER) == STEP_BY_CFL
            (self.cflRadio if byCFL else self.userTimeStepRadio).checked = True
            self.adaptSavingCheckBox.checked = node.GetParameter(CFL_ADAPT_SAVING_PARAMETER) != "false"
            self._velocityByHand = self._meshSizeByHand = False
            for parameter, spinBox, flag in ((CFL_VELOCITY_PARAMETER, self.velocitySpinBox, "_velocityByHand"),
                                             (CFL_MESH_SIZE_PARAMETER, self.meshSizeSpinBox, "_meshSizeByHand")):
                try:
                    typed = float(node.GetParameter(parameter) or "nan")
                except ValueError:
                    typed = float("nan")
                if typed == typed:     # not NaN: it was typed over
                    spinBox.value = typed
                    setattr(self, flag, True)
            try:
                self.cyclesSpinBox.value = int(node.GetParameter(CYCLES_PARAMETER) or time_step.DEFAULT_CYCLES)
            except ValueError:
                self.cyclesSpinBox.value = time_step.DEFAULT_CYCLES
            mesh = node.GetNodeReference(INPUT_MESH_REFERENCE)
            if mesh is not None:
                wasBlocked = self.ui.inputMeshSelector.blockSignals(True)
                self.ui.inputMeshSelector.setCurrentNode(mesh)
                self.ui.inputMeshSelector.blockSignals(wasBlocked)
        finally:
            self._updating = False
        self.showSettings()
        self.onMeshChanged()
        self.updateDerived()
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
        node.SetParameter(CFL_PARAMETER, repr(float(self.cflSpinBox.value)))
        # Only what was typed over: one left to the case is worked out again from it on opening.
        node.SetParameter(CFL_VELOCITY_PARAMETER, repr(float(self.velocitySpinBox.value)) if self._velocityByHand else "")
        node.SetParameter(CFL_MESH_SIZE_PARAMETER, repr(float(self.meshSizeSpinBox.value)) if self._meshSizeByHand else "")
        node.SetParameter(RUN_LENGTH_PARAMETER, RUN_BY_CYCLES if self.runByCycles() else RUN_BY_STEPS)
        node.SetParameter(TIME_STEP_MODE_PARAMETER, STEP_BY_CFL if self.timeStepByCFL() else STEP_BY_USER)
        node.SetParameter(CFL_ADAPT_SAVING_PARAMETER, "true" if self.adaptSavingCheckBox.checked else "false")
        node.SetParameter(CYCLES_PARAMETER, str(self.cyclesSpinBox.value))
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
        self.showEstimate("")
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
        # The mesh brings its size and its conditions, and with them the cycle a run is counted in.
        self.updateDerived()
        self.saveToParameterNode()
        self.updateButtons()

    # -- the time step from the CFL ---------------------------------------------------------
    def meanEdgeLength(self, node):
        """The mesh's mean edge length, in mm: measured once per mesh and change, a second or two
        on a million elements."""
        key = (node.GetID(), node.GetMesh().GetMTime())
        if self._edgeLength is None or self._edgeLength[:2] != key:
            self._edgeLength = key + (time_step.mean_edge_length(node.GetMesh()),)
        return self._edgeLength[2]

    def updateTimeStepFromCFL(self):
        """Time step size from the CFL number, the max velocity and the mesh size, and the saving
        kept in time with it; or, where the case does not yet say enough, why not, under it.

        The two that come from the case are worked out first, and put in their fields unless they
        were typed over. The estimate uses the worked-out values themselves, not their fields'
        rounding of them; a value typed over is used as typed. Worked out as the panel changes, so
        anything missing is said rather than raised.
        """
        node = self.ui.inputMeshSelector.currentNode()
        read = [key for key in (STEPS_KEY, "general.increment_in_saving_vtk_files",
                                "general.start_saving_after_time_step",
                                "general.increment_in_saving_restart_files",
                                "mesh.scale_factor", "bc.inflow_profile", "bc.impose_flux")
                if key in self.invalid]
        if read:
            self.showEstimate(" ".join(self.invalid[key][1] for key in read), warning=True)
            return None
        values = settings.resolve(self.values)
        conditions = self.conditionsTable.conditionsByName() if self.conditionsTable else {}
        named = node is not None and node.GetMesh() is not None and bool(self.conditionsTable.caps)
        areas = {name: geometry.area for _faceId, name, geometry in self.conditionsTable.caps} if named else {}

        # The case's own: the mesh's mean edge length, and the fastest velocity an inflow imposes.
        caseEdge = caseVelocity = None
        missing = []
        if named:
            try:
                with slicer.util.WaitCursor():
                    caseEdge = self.meanEdgeLength(node)
            except time_step.EstimateError as error:
                missing.append(str(error))
            try:
                if values["bc.impose_flux"]:
                    caseVelocity = time_step.fastest_inflow(conditions, areas, values["bc.inflow_profile"],
                                                            values["mesh.scale_factor"]).peak_velocity
                else:
                    missing.append(_("With Impose flux off, an inflow's values are velocities, not flows, "
                                     "and give no max velocity: type one."))
            except time_step.EstimateError as error:
                missing.append(str(error))
        else:
            missing.append(_("Select a volume mesh whose faces are named."))
        if not self._meshSizeByHand and caseEdge is not None:
            self.setSpinBox(self.meshSizeSpinBox, caseEdge)
        if not self._velocityByHand and caseVelocity is not None:
            self.setSpinBox(self.velocitySpinBox, caseVelocity)
        for label, byHand in ((self.velocityLabel, self._velocityByHand), (self.meshSizeLabel, self._meshSizeByHand)):
            font = label.font
            font.setBold(byHand)
            label.font = font

        edge = self.meshSizeSpinBox.value if self._meshSizeByHand else caseEdge
        velocity = self.velocitySpinBox.value if self._velocityByHand else caseVelocity
        if edge is None or velocity is None:
            self.showEstimate(" ".join(missing) or _("There is no mesh size or max velocity to estimate from."),
                              warning=True)
            return None
        try:
            found = time_step.estimate(self.cflSpinBox.value, edge, conditions, areas, values,
                                       cycles=self.cyclesToRun(),
                                       velocity=velocity if self._velocityByHand else None,
                                       mesh_edge_length=caseEdge,
                                       adapt_saving=bool(self.adaptSavingCheckBox.checked))
        except (time_step.EstimateError, settings.SettingError) as error:
            self.showEstimate(str(error), warning=True)
            return None
        self.setDerivedValue(TIME_STEP_KEY, found.time_step)
        # The saving, kept as it was in time: counted in steps, it would otherwise save as many
        # times more results as the step is shorter. Set by hand, it is left as it is.
        if found.saving_adapted:
            for key, value in zip(SAVING_KEYS, (found.saving.results_every, found.saving.start_after,
                                                found.saving.restart_every)):
                self.setDerivedValue(key, value)
        self.showEstimate(time_step.describe(found))
        return found

    def onConditionsChanged(self):
        self._filesDirty = True
        self.updateDerived()
        self.saveToParameterNode()
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
        self.test_aTimeStepIsEstimatedFromTheCFL()
        self.setUp()
        self.test_aSavingTypedWithTheCFLIsKeptAsTyped()
        self.setUp()
        self.test_aRunIsSetInCardiacCyclesOrInSteps()
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
        widget.cflSpinBox.value = time_step.DEFAULT_CFL
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

    def test_aTimeStepIsEstimatedFromTheCFL(self):
        """Chosen, the time step is the CFL's -- from the inflow in mL/s over its cap in mm^2 and the
        mesh's edges in mm, as the package works it out -- shown greyed, and the saving with it;
        chosen back, the step is the user's to type again."""
        widget = self.widget()
        self.readyCube(widget)
        # On screen, as a user has it, so that its buttons can be pressed: the module shown, and
        # CFD Simulation and its General group open.
        slicer.util.selectModule("SimVascularMultiPhysics")
        widget.ui.simulationCollapsibleButton.collapsed = False
        for group in widget.ui.settingsContainer.findChildren(ctk.ctkCollapsibleGroupBox):
            group.collapsed = group.title != settings.GENERAL
        self.assertFalse(widget.timeStepByCFL(), "user-defined by default")
        self.assertTrue(widget.cflInputs.isHidden(), "the estimate's inputs only when it is chosen")
        mesh = widget.ui.inputMeshSelector.currentNode().GetMesh()
        areas = {name: geometry.area for _faceId, name, geometry in widget.conditionsTable.caps}
        expected = time_step.estimate(1.0, time_step.mean_edge_length(mesh),
                                      widget.conditionsTable.conditionsByName(), areas,
                                      settings.resolve(settings.defaults()),
                                      cycles=widget.cyclesToRun())
        widget.cflRadio.checked = True
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], expected.time_step)
        self.assertEqual(widget.values["general.increment_in_saving_vtk_files"], expected.saving.results_every)
        self.assertEqual(widget.values["general.increment_in_saving_restart_files"], expected.saving.restart_every)
        self.assertTrue(widget.fields[TIME_STEP_KEY].readOnly)
        self.assertFalse(widget.labels[TIME_STEP_KEY].font.bold(), "a worked-out value is not bolded")
        text = widget.timeStepEstimateLabel.text
        self.assertIn("mL/s", text)
        self.assertIn("mm²", text)
        parameters = widget.logic.getParameterNode()
        self.assertEqual(parameters.GetParameter(TIME_STEP_MODE_PARAMETER), STEP_BY_CFL)
        saved = json.loads(parameters.GetParameter(SETTINGS_PARAMETER))
        self.assertAlmostEqual(saved[TIME_STEP_KEY], expected.time_step)

        # Filled in from the case, and not bold: nobody typed them.
        self.assertAlmostEqual(widget.velocitySpinBox.value, expected.velocity, places=2)
        self.assertAlmostEqual(widget.meshSizeSpinBox.value, time_step.mean_edge_length(mesh), places=4)
        self.assertFalse(widget.velocityLabel.font.bold())
        self.assertTrue(widget.caseValuesButton.enabled, "always clickable, typed over or not")

        # A max velocity typed over the case's is used, bold, and kept; the case's comes back on asking.
        widget.velocitySpinBox.value = 2 * expected.velocity
        typed = time_step.estimate(1.0, time_step.mean_edge_length(mesh),
                                   widget.conditionsTable.conditionsByName(), areas,
                                   settings.resolve(widget.values), cycles=widget.cyclesToRun(),
                                   velocity=2 * expected.velocity)
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], typed.time_step)
        self.assertTrue(widget.velocityLabel.font.bold())
        self.assertIn("set by hand", widget.timeStepEstimateLabel.text)
        self.assertNotEqual(parameters.GetParameter(CFL_VELOCITY_PARAMETER), "")
        widget.caseValuesButton.click()
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], expected.time_step)
        self.assertFalse(widget.velocityLabel.font.bold())
        self.assertEqual(widget.logic.getParameterNode().GetParameter(CFL_VELOCITY_PARAMETER), "")

        widget.cflSpinBox.value = 0.5
        half = time_step.estimate(0.5, time_step.mean_edge_length(mesh),
                                  widget.conditionsTable.conditionsByName(), areas,
                                  settings.resolve(widget.values), cycles=widget.cyclesToRun())
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], half.time_step,
                               msg="the CFL changed, the step worked out again")
        self.assertLess(widget.values[TIME_STEP_KEY], expected.time_step)
        widget.userTimeStepRadio.checked = True
        self.assertFalse(widget.fields[TIME_STEP_KEY].readOnly)
        self.assertTrue(widget.cflInputs.isHidden(), "the estimate's inputs only when it is chosen")
        self.delayDisplay("The time step is estimated from the CFL, or typed")

    def giveTheInletAWaveform(self, widget):
        """A 0.8 s waveform through the cube's inlet, chosen as a user chooses its file."""
        import tempfile

        from svmeshcomplete import testing

        flow = os.path.join(tempfile.mkdtemp(), "cap_inlet.flow")
        with open(flow, "w") as handle:
            handle.write("0.0 5.0\n0.4 15.0\n0.8 5.0\n")
        table = widget.conditionsTable
        table.chooseWaveformFile = lambda _directory: flow
        table.onCellDoubleClicked(table.rowOf(testing.INLET_ID), VALUES_COLUMN)
        del table.chooseWaveformFile

    def test_aSavingTypedWithTheCFLIsKeptAsTyped(self):
        """Estimated, the saving is adapted to the step; a saving setting typed is the user's, kept
        in steps as typed as the step changes and when the scene is opened again, until the
        adapting is asked for back."""
        widget = self.widget()
        self.readyCube(widget)
        self.giveTheInletAWaveform(widget)
        widget.cflRadio.checked = True
        self.assertTrue(widget.adaptSavingCheckBox.checked, "adapted by default")
        results = "general.increment_in_saving_vtk_files"
        self.assertFalse(widget.labels[results].font.bold(), "adapted, the saving is not the user's")

        mesh = widget.ui.inputMeshSelector.currentNode().GetMesh()
        areas = {name: geometry.area for _faceId, name, geometry in widget.conditionsTable.caps}

        def expected(cfl, adapt):
            return time_step.estimate(cfl, time_step.mean_edge_length(mesh),
                                      widget.conditionsTable.conditionsByName(), areas,
                                      settings.resolve(widget.values), cycles=widget.cyclesToRun(),
                                      adapt_saving=adapt)

        widget.fields[results].text = "7"
        widget.onSettingEdited(results)
        self.assertFalse(widget.adaptSavingCheckBox.checked, "typed: no longer adapted")
        self.assertTrue(widget.labels[results].font.bold())
        restart = widget.values["general.increment_in_saving_restart_files"]
        widget.cflSpinBox.value = 0.5
        self.assertEqual(widget.values[results], 7, "kept in steps as the step changes")
        self.assertEqual(widget.values["general.increment_in_saving_restart_files"], restart)
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], expected(0.5, False).time_step,
                               msg="the step made whole per cycle, with no saving to fit")
        self.assertIn("The saving is as set: results every 7 steps", widget.timeStepEstimateLabel.text)
        parameters = widget.logic.getParameterNode()
        self.assertEqual(parameters.GetParameter(CFL_ADAPT_SAVING_PARAMETER), "false")
        widget.restoreFromParameterNode()
        self.assertFalse(widget.adaptSavingCheckBox.checked)
        self.assertEqual(widget.values[results], 7, "and as typed when the scene is opened again")

        adapted = expected(0.5, True)
        widget.adaptSavingCheckBox.checked = True
        self.assertEqual(widget.values[results], adapted.saving.results_every)
        self.assertAlmostEqual(widget.values[TIME_STEP_KEY], adapted.time_step)
        widget.fields[results].text = "7"
        widget.onSettingEdited(results)
        widget.userTimeStepRadio.checked = True
        widget.cflRadio.checked = True
        self.assertTrue(widget.adaptSavingCheckBox.checked, "chosen again, the estimate starts afresh")
        widget.userTimeStepRadio.checked = True
        self.delayDisplay("A saving typed with the CFL is kept as typed")

    def test_aRunIsSetInCardiacCyclesOrInSteps(self):
        """The steps typed by default; or ten cycles of the waveform, chosen, as steps that follow
        the time step; and typed whichever is chosen when the inflow is steady, with no cycle to
        count."""
        widget = self.widget()
        self.readyCube(widget)
        self.assertFalse(widget.runByCycles(), "user-defined by default")
        self.assertEqual(widget.values[STEPS_KEY], 10000)
        self.assertFalse(widget.fields[STEPS_KEY].readOnly)
        widget.cyclesRadio.checked = True
        self.assertFalse(widget.fields[STEPS_KEY].readOnly,
                         "a steady inflow has no cycle to count: its steps are typed, cycles or not")
        self.assertTrue(widget.cyclesSpinBox.enabled, "never greyed: it says why instead")
        self.assertFalse(widget.cyclesNoteLabel.isHidden())
        self.assertIn("steady", widget.cyclesNoteLabel.text)
        self.giveTheInletAWaveform(widget)
        self.assertTrue(widget.cyclesNoteLabel.isHidden())
        self.assertEqual(widget.cyclesSpinBox.value, time_step.DEFAULT_CYCLES)
        self.assertEqual(widget.values[STEPS_KEY], 8000, "10 cycles of 0.8 s at 1 ms")
        self.assertTrue(widget.fields[STEPS_KEY].readOnly)

        field = widget.fields["general.time_step_size"]
        field.text = "2e-3"
        widget.onSettingEdited("general.time_step_size")
        self.assertEqual(widget.values[STEPS_KEY], 4000, "the steps follow the time step")

        widget.stepsRadio.checked = True
        self.assertFalse(widget.fields[STEPS_KEY].readOnly)
        widget.fields[STEPS_KEY].text = "500"
        widget.onSettingEdited(STEPS_KEY)
        self.assertEqual(widget.values[STEPS_KEY], 500)
        parameters = widget.logic.getParameterNode()
        self.assertEqual(parameters.GetParameter(RUN_LENGTH_PARAMETER), RUN_BY_STEPS)
        widget.cyclesRadio.checked = True
        self.assertEqual(widget.values[STEPS_KEY], 4000)
        # Changing the cycles with the steps user-defined chooses the cycles.
        widget.stepsRadio.checked = True
        widget.cyclesSpinBox.value = 5
        self.assertTrue(widget.runByCycles())
        self.assertEqual(widget.values[STEPS_KEY], 2000, "5 cycles of 0.8 s at 2 ms")
        self.delayDisplay("A run is set in cardiac cycles, or in steps")

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
