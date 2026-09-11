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

import csv
import os

import numpy as np
import qt
import slicer
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

# The highlight's own node, kept out of the way of anything the operator has.
HIGHLIGHT_NODE_NAME = "Mesh Prep face highlight"


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
        self._meshWasVisible = None

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

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.faceIdArrayLineEdit.connect("editingFinished()", self.onMeshChanged)
        self.ui.facesTable.connect("itemSelectionChanged()", self.onSelectionChanged)
        self.ui.facesTable.connect("cellChanged(int,int)", self.onNameEdited)
        self.ui.highlightCheckBox.connect("toggled(bool)", self.onSelectionChanged)
        self.ui.hideMeshCheckBox.connect("toggled(bool)", self.onSelectionChanged)
        self.ui.loadTableButton.connect("clicked(bool)", self.onLoadTable)
        self.ui.saveTableButton.connect("clicked(bool)", self.onSaveTable)
        self.ui.exportButton.connect("clicked(bool)", self.onExport)

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
        self.setStatus(
            _("{count} faces. {unnamed} still to name.").format(
                count=len(self._measured),
                unnamed=sum(1 for face in self._measured if not self._names.get(face.face_id)),
            )
        )
        if not self.ui.outputDirectoryPathLineEdit.currentPath:
            self.ui.outputDirectoryPathLineEdit.currentPath = self.logic.suggestedOutputDirectory(node)

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
        self.updateButtons()

    def selectedFace(self):
        rows = {index.row() for index in self.ui.facesTable.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._measured[row] if 0 <= row < len(self._measured) else None

    def onSelectionChanged(self, *_args):
        face = self.selectedFace()
        if face is None or not self.ui.highlightCheckBox.checked:
            self.clearHighlight()
            return
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or node.GetMesh() is None:
            return
        self.logic.highlight(
            node.GetMesh(), face.face_id, self.ui.faceIdArrayLineEdit.text
        )
        self.setMeshVisible(not self.ui.hideMeshCheckBox.checked)

    def clearHighlight(self):
        self.logic.clearHighlight()
        self.setMeshVisible(True)

    def setMeshVisible(self, visible):
        node = self.ui.inputMeshSelector.currentNode()
        display = node.GetDisplayNode() if node else None
        if display is None:
            return
        if self._meshWasVisible is None:
            self._meshWasVisible = display.GetVisibility()
        display.SetVisibility(self._meshWasVisible if visible else False)
        if visible:
            self._meshWasVisible = None

    # -- the face table file -----------------------------------------------
    def onLoadTable(self):
        path = self.ui.faceTablePathLineEdit.currentPath
        if not path:
            self.setStatus(_("Give the path of a face table to read."), warning=True)
            return
        try:
            names = self.logic.readNames(path)
        except (OSError, ValueError) as error:
            self.setStatus(str(error), warning=True)
            return
        unknown = sorted(set(names) - {face.face_id for face in self._measured})
        self._names.update(names)
        self.populateTable()
        message = _("Read {count} names from {name}.").format(
            count=len(names), name=os.path.basename(path)
        )
        if unknown:
            message += " " + _(
                "Faces {faces} are named in it but not in this mesh, so those names are "
                "carried but unused -- which is what a table written against a different "
                "clip looks like."
            ).format(faces=unknown)
        self.setStatus(message, warning=bool(unknown))

    def onSaveTable(self):
        path = self.ui.faceTablePathLineEdit.currentPath
        if not path:
            self.setStatus(_("Give the path to write the face table to."), warning=True)
            return
        try:
            self.logic.writeNames(path, self._measured, self._names)
        except (OSError, ValueError) as error:
            self.setStatus(str(error), warning=True)
            return
        self.setStatus(_("Wrote {path}.").format(path=path))

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
        self.ui.saveTableButton.enabled = bool(self._measured)

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
    def suggestedOutputDirectory(node):
        """The `mesh` folder beside wherever the node's scene was saved, if anywhere."""
        sceneDirectory = slicer.mrmlScene.GetRootDirectory()
        if not sceneDirectory or not os.path.isdir(sceneDirectory):
            return ""
        return os.path.join(sceneDirectory, "mesh")

    @staticmethod
    def readNames(path):
        """`{face id: name}` out of a face table, whether or not every row is filled in."""
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "FaceID" not in reader.fieldnames:
                raise ValueError(
                    _("{path} has no FaceID column, so it is not a face table.").format(path=path)
                )
            names = {}
            for row in reader:
                rawId = (row.get("FaceID") or "").strip()
                if rawId:
                    names[int(rawId)] = (row.get("Name") or "").strip()
        return names

    @staticmethod
    def writeNames(path, measured, names):
        """Write the table the command line tools read, measurements included."""
        columns = ("FaceID", "ClipVesselName", "Name", "Cells", "Area", "Diameter",
                   "X", "Y", "Z", "Flatness")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for face in measured:
                writer.writerow({
                    "FaceID": face.face_id,
                    "ClipVesselName": "",
                    "Name": names.get(face.face_id, ""),
                    "Cells": face.cell_count,
                    "Area": f"{face.area:.4f}",
                    "Diameter": f"{face.effective_diameter:.3f}",
                    "X": f"{face.centroid[0]:.2f}",
                    "Y": f"{face.centroid[1]:.2f}",
                    "Z": f"{face.centroid[2]:.2f}",
                    "Flatness": f"{face.flatness:.4f}",
                })

    def export(self, mesh, measured, names, directory, faceIdArrayNames):
        """Write the mesh-complete folder, through the package's own checks."""
        table = face_table.FaceTable([
            face_table.Face(face.face_id, names[face.face_id]) for face in measured
        ])
        arrayName = self._faceIdArrayName(mesh, faceIdArrayNames)
        return mesh_complete.write_mesh_complete(
            mesh, table, directory, face_id_array_name=arrayName
        )

    @staticmethod
    def _faceIdArrayName(mesh, faceIdArrayNames):
        for candidate in (name.strip() for name in (faceIdArrayNames or "").split(",")):
            if candidate and mesh.GetCellData().GetArray(candidate) is not None:
                return candidate
        return None

    # -- the highlight -----------------------------------------------------
    def highlight(self, mesh, faceId, faceIdArrayNames):
        """Put one face of the mesh in a node of its own, so it can be seen."""
        arrayName = self._faceIdArrayName(mesh, faceIdArrayNames)
        if arrayName is None:
            return None
        surface = faces.boundary_of(mesh)
        ids = vtk_to_numpy(surface.GetCellData().GetArray(arrayName)).astype(np.int64)
        selected = np.flatnonzero(ids == faceId)
        face = cells.as_polydata(cells.extract(surface, selected))

        node = slicer.mrmlScene.GetFirstNodeByName(HIGHLIGHT_NODE_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", HIGHLIGHT_NODE_NAME
            )
            node.CreateDefaultDisplayNodes()
            node.SetSaveWithScene(False)
            display = node.GetDisplayNode()
            display.SetColor(1.0, 0.35, 0.0)
            display.SetEdgeVisibility(True)
            display.SetLineWidth(2)
        node.SetAndObserveMesh(face)
        node.GetDisplayNode().SetVisibility(True)
        return node

    @staticmethod
    def clearHighlight():
        node = slicer.mrmlScene.GetFirstNodeByName(HIGHLIGHT_NODE_NAME)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)


class SimVascularMeshPrepTest(ScriptedLoadableModuleTest):
    """Runs under Slicer; the package's own tests run without it."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_measureNamesAndExports()

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

        table = os.path.join(directory, "face_table.csv")
        logic.writeNames(table, measured, names)
        self.assertEqual(logic.readNames(table), names)

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mesh")
        node.SetAndObserveMesh(mesh)
        highlighted = logic.highlight(mesh, wall.face_id, "CellEntityIds")
        self.assertEqual(highlighted.GetMesh().GetNumberOfCells(), wall.cell_count)
        logic.clearHighlight()
        self.assertIsNone(slicer.mrmlScene.GetFirstNodeByName(HIGHLIGHT_NODE_NAME))

        self.delayDisplay("Measured, named, exported and highlighted")
