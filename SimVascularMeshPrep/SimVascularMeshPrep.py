"""Name the faces of a volume mesh and write the folder an svMultiPhysics case reads.

A mesher hands back one grid holding the volume elements and the boundary cells they stand
on, every cell carrying a face id. svMultiPhysics reads a folder: the volume elements alone
under `GlobalNodeID`/`GlobalElementID`, the exterior under `ModelFaceID`, and one file per
named face, which is what its boundary conditions bind to. No mesher writes any of that.

Between the two sits a step nothing can do for you. A face id is a number; a boundary
condition is per vessel. Somebody has to say which cap is the azygous vein -- and the names
they give become the `mesh-surfaces/` file names, and through them the `Add_face` and
`Add_BC` names in `solver.xml`, so they are what every result comes back labelled with.
That is what this panel is for: the faces measured and listed, the selected one shown in
the 3D view, a name typed against it.

## Where the geometry is

All of it is in `svmeshcomplete`, the package beside this file, which imports nothing from
Slicer and is pip-installable on its own. This file is an MRML adapter: it reads the
selected node's grid, calls the package, and puts what it says into a table.

The split is what keeps a case prepared here and a case prepared from a terminal identical
-- the workflow scripts that package Fontan cases call the same functions -- and it is why
the checks below are the package's rather than this panel's. All three of them catch inputs
that otherwise reach the solver and are misread in silence: a volume mesh of more than one
element type, a face table that does not describe the mesh, and a boundary cell standing
against no volume element.

## Flatness

The one measurement worth explaining. It is the largest distance of any point of a face
from that face's own best-fit plane, so a cap cut normal to its vessel reads 0 and one that
was not cut cleanly does not. The flow crossing a cap that is not planar is not what its
boundary condition says, and nothing downstream will notice.
"""

import json
import logging
import os

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from svmeshcomplete import cells, face_table, faces, mesh_complete

# Columns of the table, and which of them the operator edits.
COLUMNS = ("Face", "Name", "Cells", "Area", "Diameter", "Flatness")
NAME_COLUMN = COLUMNS.index("Name")

# Where the panel's state is kept so that it is saved with the scene, which is what makes
# the naming survive closing Slicer. The names are the expensive part -- twenty-odd caps
# matched to vessels by eye -- and a scene that came back without them would be a scene
# whose mesh had to be named again.
NAMES_PARAMETER = "FaceNames"
FACE_ID_ARRAY_PARAMETER = "FaceIdArrayNames"
OUTPUT_DIRECTORY_PARAMETER = "OutputDirectory"
INPUT_MESH_REFERENCE = "InputMesh"

# The highlight's own node, kept out of the way of anything the operator has, and the
# colour it is drawn in: yellow against the grey the mesh sits at.
HIGHLIGHT_NODE_NAME = "Mesh Prep face highlight"
HIGHLIGHT_COLOR = (1.0, 1.0, 0.0)

# What the mesh is left as when face colouring is turned off: the same neutral grey Clip
# Vessel gives its output, rather than whatever colour the node happened to be created
# with, and see-through enough to find a cap behind it without reaching for a button.
SOLID_COLOR = (0.75, 0.75, 0.75)
SOLID_OPACITY = 0.8

# What the transparency button drops to, and comes back from.
TRANSPARENT_OPACITY = 0.4


class SimVascularMeshPrep(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SimVascular Mesh Prep")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SimVascular")]
        self.parent.dependencies = []
        self.parent.contributors = ["Aaron Brown (Stanford University)"]
        self.parent.helpText = _(
            "Name the faces of a volume mesh and write the mesh-complete folder an "
            "svMultiPhysics case reads. The mesh is one carrying a face id per cell, "
            "typically out of CFD Mesh Generator; the output is the volume elements, the "
            "exterior surface and one file per named face, under the GlobalNodeID the "
            "solver binds its boundary conditions through."
        )
        self.parent.acknowledgementText = _(
            "Developed in the Cardiovascular Biomechanics Computation Lab at Stanford "
            "University. The mesh packaging is the svmeshcomplete package beside this "
            "module, which runs outside Slicer as well."
        )


class SimVascularMeshPrepWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._measured = []
        self._names = {}
        self._updating = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimVascularMeshPrep.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimVascularMeshPrepLogic()

        table = self.ui.facesTable
        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels([_(name) for name in COLUMNS])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(NAME_COLUMN, qt.QHeaderView.Stretch)

        # The three buttons beside the selector, none of them checkable: a mark would be
        # saying what the display node holds, and nothing tells this when that changes
        # elsewhere, so it would sooner or later contradict the scene. Each reads the state
        # at the moment it is pressed and turns it around.
        rowHeight = self.ui.inputMeshSelector.sizeHint.height()
        buttons = (
            (self.ui.meshEdgesButton, self.resourcePath("Icons/ToggleEdges.svg"),
             self.onToggleEdges),
            (self.ui.meshColorsButton, self.resourcePath("Icons/ToggleFaceColors.svg"),
             self.onToggleFaceColors),
            (self.ui.meshTransparencyButton, self.resourcePath("Icons/ToggleTransparency.svg"),
             self.onToggleTransparency),
            (self.ui.meshVisibilityButton, ":/Icons/Medium/SlicerVisibleInvisible.png",
             self.onToggleVisibility),
        )
        for button, icon, handler in buttons:
            button.setIcon(qt.QIcon(icon))
            button.setAutoRaise(True)
            button.setFixedHeight(rowHeight)
            button.setIconSize(qt.QSize(rowHeight - 8, rowHeight - 8))
            button.connect("clicked()", handler)

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.faceIdArrayLineEdit.connect("editingFinished()", self.onMeshChanged)
        self.ui.facesTable.connect("itemSelectionChanged()", self.onSelectionChanged)
        self.ui.facesTable.connect("cellChanged(int,int)", self.onNameEdited)
        self.ui.exportButton.connect("clicked(bool)", self.onExport)
        self.ui.outputDirectoryPathLineEdit.connect(
            "currentPathChanged(QString)", self.onOutputDirectoryChanged
        )

        # A scene being loaded replaces the parameter node, so the panel has to read itself
        # back out of the new one rather than trust what it is showing.
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        self.restoreFromParameterNode()
        self.updateButtons()

    def enter(self):
        """Coming back to the panel after a scene was loaded elsewhere in the application."""
        self.restoreFromParameterNode()

    def onSceneEndImport(self, caller=None, event=None):
        self.restoreFromParameterNode()

    def onSceneEndClose(self, caller=None, event=None):
        self._measured = []
        self._names = {}
        self.populateTable()

    # -- state that is saved with the scene ---------------------------------
    def parameterNode(self):
        return self.logic.getParameterNode()

    def restoreFromParameterNode(self):
        """Put the panel back the way the scene left it."""
        node = self.parameterNode()
        if node is None:
            return
        self._updating = True
        try:
            self._names = self.logic.readNamesParameter(node.GetParameter(NAMES_PARAMETER))
            arrayNames = node.GetParameter(FACE_ID_ARRAY_PARAMETER)
            if arrayNames:
                self.ui.faceIdArrayLineEdit.text = arrayNames
            directory = node.GetParameter(OUTPUT_DIRECTORY_PARAMETER)
            if directory:
                self.ui.outputDirectoryPathLineEdit.currentPath = directory
            mesh = node.GetNodeReference(INPUT_MESH_REFERENCE)
            if mesh is not None:
                self.ui.inputMeshSelector.setCurrentNode(mesh)
        finally:
            self._updating = False
        self.onMeshChanged()

    def saveToParameterNode(self):
        node = self.parameterNode()
        if node is None:
            return
        was_modifying = node.StartModify()
        node.SetParameter(NAMES_PARAMETER, self.logic.writeNamesParameter(self._names))
        node.SetParameter(FACE_ID_ARRAY_PARAMETER, self.ui.faceIdArrayLineEdit.text)
        node.SetParameter(
            OUTPUT_DIRECTORY_PARAMETER, self.ui.outputDirectoryPathLineEdit.currentPath or ""
        )
        node.SetNodeReferenceID(
            INPUT_MESH_REFERENCE,
            self.ui.inputMeshSelector.currentNodeID or None,
        )
        node.EndModify(was_modifying)

    def onOutputDirectoryChanged(self, *_args):
        if not self._updating:
            self.saveToParameterNode()
            self.updateButtons()

    def cleanup(self):
        self.clearHighlight()
        self.removeObservers()

    # -- reading the mesh --------------------------------------------------
    def onMeshChanged(self, _node=None):
        """Measure the selected mesh's faces, keeping any names already typed."""
        self._measured = []
        node = self.ui.inputMeshSelector.currentNode()
        mesh = node.GetMesh() if node else None
        if mesh is None:
            self.setStatus(
                _("Select a volume mesh. A model node holding a surface is not one: this "
                  "wants the grid a mesher filled it with.")
                if node else ""
            )
            self.populateTable()
            return

        try:
            self._measured = self.logic.measure(mesh, self.ui.faceIdArrayLineEdit.text)
        except ValueError as error:
            self.setStatus(str(error), warning=True)
            self.populateTable()
            return

        for face in self._measured:
            self._names.setdefault(face.face_id, "")
        self.populateTable()
        self.saveToParameterNode()
        self.updateStatus()
        if not self.ui.outputDirectoryPathLineEdit.currentPath:
            self.ui.outputDirectoryPathLineEdit.currentPath = self.logic.suggestedOutputDirectory()

    def populateTable(self):
        table = self.ui.facesTable
        # The face with the most cells is the wall -- it has far more than any cap -- and
        # is the one face flatness says nothing about.
        wall = max(self._measured, key=lambda face: face.cell_count, default=None)
        self._updating = True
        try:
            table.setRowCount(len(self._measured))
            for row, face in enumerate(self._measured):
                values = (
                    str(face.face_id),
                    self._names.get(face.face_id, ""),
                    f"{face.cell_count:,}",
                    f"{face.area:.2f}",
                    f"{face.effective_diameter:.2f}",
                    f"{face.flatness:.4f}",
                )
                for column, value in enumerate(values):
                    item = table.item(row, column) or qt.QTableWidgetItem()
                    item.setText(value)
                    if column == NAME_COLUMN:
                        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)
                    else:
                        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                    if column == COLUMNS.index("Flatness") and self.logic.looksUnplanar(
                        face, face is wall
                    ):
                        item.setToolTip(
                            _("Not planar, so this was not cut cleanly. Whatever this face "
                              "is named, the flow crossing it is not what a boundary "
                              "condition on it would say.")
                        )
                        item.setForeground(qt.QBrush(qt.QColor(200, 120, 0)))
                    table.setItem(row, column, item)
        finally:
            self._updating = False
        self.updateButtons()

    # -- naming ------------------------------------------------------------
    def onNameEdited(self, row, column):
        if self._updating or column != NAME_COLUMN:
            return
        item = self.ui.facesTable.item(row, column)
        self._names[self._measured[row].face_id] = (item.text() or "").strip()
        self.saveToParameterNode()
        self.updateStatus()
        self.updateButtons()

    def selectedFace(self):
        rows = {index.row() for index in self.ui.facesTable.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._measured[row] if 0 <= row < len(self._measured) else None

    def onSelectionChanged(self, *_args):
        """Show whichever face is selected. Always: a row on its own says very little."""
        face = self.selectedFace()
        node = self.ui.inputMeshSelector.currentNode()
        if face is None or node is None or node.GetMesh() is None:
            self.clearHighlight()
            return
        display = node.GetDisplayNode()
        self.logic.highlight(
            node.GetMesh(),
            face.face_id,
            self.ui.faceIdArrayLineEdit.text,
            showEdges=bool(display and display.GetEdgeVisibility()),
        )

    def clearHighlight(self):
        self.logic.clearHighlight()

    # -- looking at the mesh -----------------------------------------------
    def meshDisplayNode(self):
        node = self.ui.inputMeshSelector.currentNode()
        if node is None:
            return None
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        return node.GetDisplayNode()

    def onToggleVisibility(self):
        display = self.meshDisplayNode()
        if display is not None:
            display.SetVisibility(not display.GetVisibility())

    def onToggleEdges(self):
        """On the mesh and on the highlighted face together: they are one picture."""
        display = self.meshDisplayNode()
        if display is None:
            return
        visible = not display.GetEdgeVisibility()
        display.SetEdgeVisibility(visible)
        self.logic.setHighlightEdgeVisibility(visible)

    def onToggleFaceColors(self):
        display = self.meshDisplayNode()
        node = self.ui.inputMeshSelector.currentNode()
        if display is None or node is None or node.GetMesh() is None:
            return
        if display.GetScalarVisibility():
            self.logic.showSolidColor(display)
            return
        arrayName = self.logic.faceIdArrayName(
            node.GetMesh(), self.ui.faceIdArrayLineEdit.text
        )
        if arrayName is None:
            self.setStatus(
                _("Nothing to colour by: this mesh carries no face ids."), warning=True
            )
            return
        self.logic.colourByFaceIds(display, arrayName)

    def onToggleTransparency(self):
        """Between see-through and the mesh's own opacity, not between that and opaque.

        The grey the mesh sits at is already a little transparent, so testing against 1.0
        would make this button turn transparency *off* the first time it is pressed.
        """
        display = self.meshDisplayNode()
        if display is None:
            return
        transparent = display.GetOpacity() <= TRANSPARENT_OPACITY
        display.SetOpacity(SOLID_OPACITY if transparent else TRANSPARENT_OPACITY)

    # -- export ------------------------------------------------------------
    def onExport(self):
        node = self.ui.inputMeshSelector.currentNode()
        directory = self.ui.outputDirectoryPathLineEdit.currentPath
        if node is None or node.GetMesh() is None or not directory:
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to export the mesh."), waitCursor=True):
            result = self.logic.export(
                node.GetMesh(),
                self._measured,
                self._names,
                directory,
                self.ui.faceIdArrayLineEdit.text,
            )
        self.setStatus(
            _("Wrote {directory}: {elements:,} {kind} elements over {nodes:,} nodes, "
              "{faces} faces.").format(
                directory=directory,
                elements=result.number_of_elements,
                kind=result.element_type,
                nodes=result.number_of_nodes,
                faces=len(result.face_surfaces),
            )
        )

    # -- panel state -------------------------------------------------------
    def updateButtons(self):
        named = self._measured and all(
            self._names.get(face.face_id) for face in self._measured
        )
        self.ui.exportButton.enabled = bool(
            named and self.ui.outputDirectoryPathLineEdit.currentPath
        )
        self.ui.exportButton.toolTip = (
            _("Write the mesh-complete folder.")
            if named
            else _("Every face has to be named first: the names are the solver's boundary "
                   "condition names, so a face without one has nothing to bind to.")
        )

    def updateStatus(self):
        """What is left to do, which changes with every name typed.

        Not folded into updateButtons: that also runs when the output folder changes, and
        it would wipe out what an export had just reported.
        """
        self.setStatus(self.logic.statusText(self._measured, self._names))

    def setStatus(self, text, warning=False):
        self.ui.statusLabel.text = text
        self.ui.statusLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""


class SimVascularMeshPrepLogic(ScriptedLoadableModuleLogic):
    """The MRML-free work, which is `svmeshcomplete` plus the highlight node."""

    # Beyond this, in the mesh's own units, a face is not flat enough to be a clean cut.
    # A cap cut normal to its vessel comes out two orders of magnitude below it.
    FLATNESS_TOLERANCE = 0.5

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def measure(mesh, faceIdArrayNames):
        """Every labelled face of the mesh, measured."""
        names = [name.strip() for name in (faceIdArrayNames or "").split(",") if name.strip()]
        arrayName = None
        for candidate in names:
            if mesh.GetCellData().GetArray(candidate) is not None:
                arrayName = candidate
                break
        if arrayName is None and names:
            raise ValueError(
                _("The mesh carries none of {names} as face ids. Without them there is no "
                  "telling a cap from the wall; in CFD Mesh Generator the array is named "
                  "by its own 'Face ids array' field.").format(names=names)
            )
        return faces.measure_faces(mesh, arrayName)

    @classmethod
    def looksUnplanar(cls, face, isWall):
        """Whether a face is too far from its own plane to be a clean cut.

        Never asked of the wall, which is not meant to be planar and would read large on
        any anatomy at all.
        """
        return not isWall and face.flatness > cls.FLATNESS_TOLERANCE

    @staticmethod
    def suggestedOutputDirectory():
        """`mesh` beside the scene file, once the scene has one.

        Nothing is suggested for a scene that has never been saved. `GetRootDirectory`
        answers even then -- with Slicer's default, the user's Documents folder -- and a
        mesh-complete folder written there is not where anyone wanted it, least of all
        with Export enabled and looking ready.
        """
        url = slicer.mrmlScene.GetURL()
        directory = os.path.dirname(url) if url else ""
        return os.path.join(directory, "mesh") if os.path.isdir(directory) else ""

    @staticmethod
    def writeNamesParameter(names):
        """The names as one string, for a parameter node to carry into a saved scene."""
        return json.dumps(
            {str(face_id): name for face_id, name in sorted(names.items())},
            separators=(",", ":"),
        )

    @staticmethod
    def readNamesParameter(text):
        """The names back out, and nothing at all rather than an exception from a bad one."""
        if not text:
            return {}
        try:
            stored = json.loads(text)
        except ValueError:
            logging.warning("Mesh Prep could not read the face names saved with this scene.")
            return {}
        names = {}
        for face_id, name in stored.items():
            try:
                names[int(face_id)] = str(name)
            except (TypeError, ValueError):
                continue
        return names

    def export(self, mesh, measured, names, directory, faceIdArrayNames):
        """Write the mesh-complete folder, through the package's own checks.

        The names are not written beside it. They are saved with the scene, which is
        where anything needing them reads them from -- including the workflow scripts that
        package cases from a terminal. A copy in the folder would be a second answer to
        the same question, and the one that goes stale is the one on disk.
        """
        table = face_table.FaceTable([
            face_table.Face(face.face_id, names[face.face_id]) for face in measured
        ])
        arrayName = self.faceIdArrayName(mesh, faceIdArrayNames)
        return mesh_complete.write_mesh_complete(
            mesh, table, directory, face_id_array_name=arrayName
        )

    @staticmethod
    def statusText(measured, names):
        """How many faces there are and how many are still to name."""
        if not measured:
            return ""
        unnamed = sum(1 for face in measured if not names.get(face.face_id))
        if not unnamed:
            return _("{count} faces, all named.").format(count=len(measured))
        return _("{count} faces. {unnamed} still to name.").format(
            count=len(measured), unnamed=unnamed
        )

    @staticmethod
    def faceIdArrayName(mesh, faceIdArrayNames):
        """The first of the offered names the mesh carries, or None."""
        for candidate in (name.strip() for name in (faceIdArrayNames or "").split(",")):
            if candidate and mesh.GetCellData().GetArray(candidate) is not None:
                return candidate
        return None

    @staticmethod
    def showSolidColor(display):
        """Turn face colouring off, leaving the mesh a slightly see-through neutral grey."""
        display.SetScalarVisibility(False)
        display.SetColor(*SOLID_COLOR)
        display.SetOpacity(SOLID_OPACITY)

    @staticmethod
    def colourByFaceIds(display, arrayName):
        """Colour a model by its face ids, each face its own flat colour.

        The ids are labels rather than a measurement, so the colour table has to be one
        whose neighbouring entries differ -- a continuous one would give twenty pulmonary
        branches twenty shades of the same colour.
        """
        display.SetActiveScalarName(arrayName)
        display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
        for colourNodeId in ("vtkMRMLColorTableNodeRandom", "vtkMRMLColorTableNodeLabels"):
            if slicer.mrmlScene.GetNodeByID(colourNodeId) is not None:
                display.SetAndObserveColorNodeID(colourNodeId)
                break
        display.SetScalarRangeFlag(display.UseDataScalarRange)
        display.SetScalarVisibility(True)

    # -- the highlight -----------------------------------------------------
    def highlight(self, mesh, faceId, faceIdArrayNames, showEdges=False):
        """Put one face of the mesh in a node of its own, so it can be seen.

        The node is not saved with the scene and is hidden from the editors, so it stays
        out of the subject hierarchy and out of every node selector: it is a way of
        looking at the mesh, not a thing the case has. Hidden before it is added, because
        the subject hierarchy takes its item from a node as it arrives.
        """
        arrayName = self.faceIdArrayName(mesh, faceIdArrayNames)
        if arrayName is None:
            return None
        surface = faces.boundary_of(mesh)
        ids = vtk_to_numpy(surface.GetCellData().GetArray(arrayName)).astype(np.int64)
        selected = np.flatnonzero(ids == faceId)
        face = cells.as_polydata(cells.extract(surface, selected))

        node = self.highlightNode()
        if node is None:
            node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelNode")
            node.SetName(HIGHLIGHT_NODE_NAME)
            node.SetSaveWithScene(False)
            node.SetHideFromEditors(True)
            node = slicer.mrmlScene.AddNode(node)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSaveWithScene(False)
            node.GetDisplayNode().SetHideFromEditors(True)
        node.SetAndObserveMesh(face)

        # Set every time rather than only on the node's first use, so that a highlight left
        # over from an earlier run of the module cannot keep an earlier appearance.
        display = node.GetDisplayNode()
        display.SetColor(*HIGHLIGHT_COLOR)
        display.SetLineWidth(2)
        # One colour whichever way a cell faces. Slicer shifts a backface's hue so that
        # inside can be told from outside -- by (-0.05, -0.1, 0), which turns this yellow
        # into RGB (1, 0.73, 0.1), an orange. On a cap that is the strips the boundary
        # layer sweeps out at the vessel end, wound the other way from the cap's own
        # triangles, and a highlight saying "this is the face" has nothing to say about
        # which way round its cells are.
        display.SetBackfaceColorHSVOffset(0.0, 0.0, 0.0)
        display.SetEdgeVisibility(showEdges)
        display.SetVisibility(True)
        return node

    @staticmethod
    def highlightNode():
        """The highlight's node, which being hidden is not found by name lookups."""
        for index in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = slicer.mrmlScene.GetNthNodeByClass(index, "vtkMRMLModelNode")
            if node.GetName() == HIGHLIGHT_NODE_NAME:
                return node
        return None

    @classmethod
    def setHighlightEdgeVisibility(cls, visible):
        node = cls.highlightNode()
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetEdgeVisibility(visible)

    @classmethod
    def clearHighlight(cls):
        node = cls.highlightNode()
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)


class SimVascularMeshPrepTest(ScriptedLoadableModuleTest):
    """Runs under Slicer; the package's own tests run without it."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_measureNamesAndExports()
        self.setUp()
        self.test_namesSurviveASavedScene()

    def test_measureNamesAndExports(self):
        import tempfile

        from svmeshcomplete import testing

        logic = SimVascularMeshPrepLogic()
        mesh = testing.cube_mesh()
        measured = logic.measure(mesh, "CellEntityIds, ModelFaceID")
        self.assertEqual(len(measured), 3)

        wall = max(measured, key=lambda face: face.cell_count)
        names = {
            face.face_id: "wall" if face is wall else f"cap_{face.face_id}"
            for face in measured
        }

        directory = tempfile.mkdtemp()
        result = logic.export(mesh, measured, names, directory, "CellEntityIds")
        self.assertEqual(result.element_type, "Tetra")
        self.assertEqual(len(result.face_surfaces), 3)
        self.assertTrue(os.path.isfile(os.path.join(directory, mesh_complete.VOLUME_MESH_NAME)))

        # The names are the scene's, not the folder's.
        self.assertFalse(os.path.isfile(os.path.join(directory, "face_table.csv")))

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mesh")
        node.SetAndObserveMesh(mesh)
        highlighted = logic.highlight(mesh, wall.face_id, "CellEntityIds", showEdges=True)
        self.assertEqual(highlighted.GetMesh().GetNumberOfCells(), wall.cell_count)
        self.assertTrue(highlighted.GetDisplayNode().GetEdgeVisibility())
        self.assertEqual(
            tuple(round(c, 2) for c in highlighted.GetDisplayNode().GetColor()),
            HIGHLIGHT_COLOR,
        )
        # A backface has to be the same colour: Slicer shifts its hue by default, which
        # turns this yellow orange on the strips a boundary layer leaves at a vessel end.
        self.assertEqual(
            tuple(highlighted.GetDisplayNode().GetBackfaceColorHSVOffset()), (0.0, 0.0, 0.0)
        )
        self.assertTrue(highlighted.GetHideFromEditors())
        self.assertFalse(highlighted.GetSaveWithScene())

        # The edges toggle reaches it.
        logic.setHighlightEdgeVisibility(False)
        self.assertFalse(highlighted.GetDisplayNode().GetEdgeVisibility())

        logic.clearHighlight()
        self.assertIsNone(logic.highlightNode())

        # A scene with nowhere to write to suggests nowhere, rather than Documents.
        self.assertEqual(logic.suggestedOutputDirectory(), "")

        # The count has to follow the naming: it is read after every name typed.
        self.assertEqual(logic.statusText([], {}), "")
        self.assertIn("3 still to name", logic.statusText(measured, {}))
        partly = {wall.face_id: "wall"}
        self.assertIn("2 still to name", logic.statusText(measured, partly))
        blank = dict(names, **{wall.face_id: ""})
        self.assertIn("1 still to name", logic.statusText(measured, blank))
        self.assertIn("all named", logic.statusText(measured, names))

        # The three display toggles, each reading the state it is turning around.
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        self.assertEqual(logic.faceIdArrayName(mesh, "CellEntityIds, ModelFaceID"), "CellEntityIds")
        logic.colourByFaceIds(display, "CellEntityIds")
        self.assertTrue(display.GetScalarVisibility())
        self.assertEqual(display.GetActiveScalarName(), "CellEntityIds")
        logic.showSolidColor(display)
        self.assertFalse(display.GetScalarVisibility())
        self.assertEqual(tuple(round(c, 2) for c in display.GetColor()), SOLID_COLOR)
        self.assertAlmostEqual(display.GetOpacity(), SOLID_OPACITY)

        display.SetEdgeVisibility(not display.GetEdgeVisibility())
        self.assertTrue(display.GetEdgeVisibility())

        self.delayDisplay("Measured, named, exported, highlighted and coloured")

    def test_namesSurviveASavedScene(self):
        """The naming is the expensive part, so it has to come back with the scene."""
        import tempfile

        from svmeshcomplete import testing

        logic = SimVascularMeshPrepLogic()
        names = {1: "wall", 2: "cap_RSVC", 3: "cap_lpa_a"}

        mesh = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "volume mesh")
        mesh.SetAndObserveMesh(testing.cube_mesh())

        parameters = logic.getParameterNode()
        parameters.SetParameter(NAMES_PARAMETER, logic.writeNamesParameter(names))
        parameters.SetParameter(FACE_ID_ARRAY_PARAMETER, "CellEntityIds, ModelFaceID")
        parameters.SetNodeReferenceID(INPUT_MESH_REFERENCE, mesh.GetID())

        scene = os.path.join(tempfile.mkdtemp(), "scene.mrb")
        self.assertTrue(slicer.util.saveScene(scene))

        # Clear(1) removes the singletons too, so this looks like a fresh application
        # rather than one that still had the parameter node in it.
        slicer.mrmlScene.Clear(1)
        self.assertEqual(
            logic.readNamesParameter(logic.getParameterNode().GetParameter(NAMES_PARAMETER)), {}
        )

        self.assertTrue(slicer.util.loadScene(scene))
        restored = logic.getParameterNode()
        self.assertEqual(
            logic.readNamesParameter(restored.GetParameter(NAMES_PARAMETER)), names
        )
        self.assertEqual(
            restored.GetParameter(FACE_ID_ARRAY_PARAMETER), "CellEntityIds, ModelFaceID"
        )
        self.assertIsNotNone(restored.GetNodeReference(INPUT_MESH_REFERENCE))

        # A parameter that is not readable loses the names rather than the panel.
        self.assertEqual(logic.readNamesParameter("not json"), {})

        self.delayDisplay("Names survived a saved scene")
