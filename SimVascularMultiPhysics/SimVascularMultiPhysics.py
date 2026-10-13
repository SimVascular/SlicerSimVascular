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
"""

import json
import logging
import os

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
from svmpsetup import case, settings
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

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimVascularMultiPhysics.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimVascularMultiPhysicsLogic()
        self.conditionsTable = BoundaryConditionsTable(
            self.ui.facesTable, self.ui.valuesHintLabel, self.setStatus,
            changed=self.updateButtons,
            startDirectory=lambda: self.ui.outputDirectoryPathLineEdit.currentPath)
        self.buildSettings()

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.openMeshPrepButton.connect("clicked(bool)", self.onOpenMeshPrep)
        self.ui.outputDirectoryPathLineEdit.connect("currentPathChanged(QString)",
                                                    self.onOutputDirectoryChanged)
        self.ui.resetSettingsButton.connect("clicked(bool)", self.onResetSettings)
        self.ui.createSolverFilesButton.connect("clicked(bool)", self.onCreateSolverFiles)

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
        self.styleSetting(key)
        self.saveToParameterNode()
        self.updateButtons()

    def onResetSettings(self):
        self.values = settings.defaults()
        self.invalid = {}
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
        self.saveToParameterNode()
        self.updateButtons()

    # -- the mesh -------------------------------------------------------------------
    def onMeshChanged(self, _node=None):
        """Read the selected mesh's names, measure its caps, and bring back its conditions."""
        self.ui.openMeshPrepButton.visible = False
        node = self.ui.inputMeshSelector.currentNode()
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

    def setStatus(self, text, warning=False):
        self.ui.statusLabel.text = text
        self.ui.statusLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""


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
